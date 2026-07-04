// PageRank benchmark with ricochet-managed graph adjacency data for gem5.
//
// Expects the ligra binary graph format (produced by adjToBinary):
//   <prefix>.config  — text: n (vertex count)
//   <prefix>.idx     — n uint32_t CSR offsets
//   <prefix>.adj     — m uint32_t neighbor IDs  [backend-managed]
//
// Flow (ricochet backend):
//   KVM:  run -warmup iterations (UFFD serves page faults), then checkpoint
//   O3:   each OMP thread registers for UINTR, runs -measure iterations,
//         then unregisters before any cross-thread synchronisation.
// Flow (mmap backend): the .adj file is mmap'd and the kernel serves faults;
//   warmup + checkpoint + measure run the same PageRank kernel with no UINTR.
//
// Usage: pagerank_ricochet <prefix> [-backend ricochet|mmap]
//                          [-warmup N] [-measure M] [-threads T] [-phys MB]

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <stdint.h>

#include <gem5/m5ops.h>
#include <m5_mmap.h>
#include "api.hpp"
#include "replacement.hpp"

static const double kDamping = 0.85;

// Heartbeat print usable inside the UIF=1 measured region: a single write()
// syscall (no stdio locks that a UINTR could interrupt mid-hold), one short
// line so concurrent per-thread writes don't tear.
static inline void progress(int tid, uint64_t done, uint64_t total) {
    char b[80];
    int l = snprintf(b, sizeof b, "[pagerank]   t%d %llu/%llu\n", tid,
                     (unsigned long long)done, (unsigned long long)total);
    (void)!write(1, b, l);
}

// Backend under test: our ricochet userspace page cache, or a plain kernel
// mmap of the .adj file (the baseline we want to beat).  Same binary, chosen
// at runtime with -backend, so both paths share the identical PageRank kernel.
enum Backend { BK_RICOCHET, BK_MMAP };

// Replacement policy for the ricochet backend:
//   default  — ricochet's built-in S3-FIFO cache (oblivious; thrashes on a
//              looping sequential scan larger than the cache).
//   pin      — app-managed scan-resistant policy: each worker pins the first
//              pages of its contiguous adj[] slice (kept resident across
//              iterations) and evict-behind for the rest.  ricochet only fills;
//              the app owns eviction (see api.hpp region_set_app_managed).
enum Policy { POL_DEFAULT, POL_PIN };

// --- app-managed scan-resistant policy (per-thread, lock-free) ---
// PageRank workers stream disjoint contiguous slices of adj[], so each thread's
// residency is independent — no cross-thread synchronization needed.
static thread_local uint64_t tl_pin_end_pg     = 0;      // pages < this are pinned
static thread_local uint64_t tl_prev_stream_pg = ~0ULL;  // last non-pinned page
static thread_local size_t   tl_evict_buf[64];
static thread_local int      tl_evict_n = 0;

// Called by ricochet after it fills a faulted page.  Pinned pages stay; for a
// streaming page we drop the previous one (the sequential scan is done with it).
static void pin_on_fault(ricochet::RicochetRegion *r, size_t offset, void *) {
    uint64_t pg = offset >> 12;
    if (pg < tl_pin_end_pg) return;                        // pinned: keep resident
    if (tl_prev_stream_pg != ~0ULL) {
        tl_evict_buf[tl_evict_n++] = (size_t)(tl_prev_stream_pg << 12);
        if (tl_evict_n == 64) {
            ricochet::do_evict_pages(r, tl_evict_buf, tl_evict_n);
            tl_evict_n = 0;
        }
    }
    tl_prev_stream_pg = pg;
}

static void pin_flush(ricochet::RicochetRegion *r) {
    if (tl_evict_n) { ricochet::do_evict_pages(r, tl_evict_buf, tl_evict_n); tl_evict_n = 0; }
    tl_prev_stream_pg = ~0ULL;
}

// Drop every PTE in the region so the app-managed measured phase starts from an
// empty residency and every first access faults through the policy hook.
static void cold_evict_region(ricochet::RicochetRegion *r) {
    size_t npages = r->size / 4096;
    size_t buf[512];
    for (size_t base = 0; base < npages; ) {
        int k = 0;
        while (k < 512 && base < npages) buf[k++] = (base++) * 4096;
        ricochet::do_evict_pages(r, buf, k);
    }
}

// Pull-based PageRank iteration (symmetric graphs: in-nbrs == out-nbrs).
// adj[] is ricochet-managed; offsets, p_curr, p_next are in normal memory.
// Safe to call with OMP barriers (use_uffd mode or UIF=0).
static void pagerank_iter_parallel(const uint32_t *adj, const uint32_t *offsets,
                                   uint64_t n, uint64_t m,
                                   const double *p_curr, double *p_next,
                                   int nthreads) {
    double add_const = (1.0 - kDamping) / (double)n;
    #pragma omp parallel for num_threads(nthreads) schedule(static)
    for (uint64_t v = 0; v < n; v++) {
        uint32_t start = offsets[v];
        uint32_t end   = (v + 1 < n) ? offsets[v + 1] : (uint32_t)m;
        double sum = 0.0;
        for (uint32_t j = start; j < end; j++) {
            uint32_t u     = adj[j];
            uint32_t u_s   = offsets[u];
            uint32_t u_e   = (u + 1 < n) ? offsets[u + 1] : (uint32_t)m;
            uint32_t u_deg = u_e - u_s;
            if (u_deg > 0)
                sum += p_curr[u] / (double)u_deg;
        }
        p_next[v] = kDamping * sum + add_const;
    }
}

// Pull-based iteration inside a parallel region with UIF=1.
// Must be called from within an omp parallel region where each thread has
// already called region_enable_uintr().  Uses manual range split — no OMP
// barriers while UIF=1 to avoid UINTR disrupting futex_wait.
static void pagerank_iter_upf(const uint32_t *adj, const uint32_t *offsets,
                               uint64_t n, uint64_t m,
                               const double *p_curr, double *p_next) {
    int tid      = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    uint64_t my_start = (uint64_t)tid * n / (uint64_t)nthreads;
    uint64_t my_end   = (uint64_t)(tid + 1) * n / (uint64_t)nthreads;
    double add_const  = (1.0 - kDamping) / (double)n;

    uint64_t span = my_end - my_start;
    uint64_t step = span / 20 ? span / 20 : 1;   // ~20 heartbeats per thread

    for (uint64_t v = my_start; v < my_end; v++) {
        if ((v - my_start) % step == 0) progress(tid, v - my_start, span);
        uint32_t start = offsets[v];
        uint32_t end   = (v + 1 < n) ? offsets[v + 1] : (uint32_t)m;
        double sum = 0.0;
        for (uint32_t j = start; j < end; j++) {
            uint32_t u     = adj[j];
            uint32_t u_s   = offsets[u];
            uint32_t u_e   = (u + 1 < n) ? offsets[u + 1] : (uint32_t)m;
            uint32_t u_deg = u_e - u_s;
            if (u_deg > 0)
                sum += p_curr[u] / (double)u_deg;
        }
        p_next[v] = kDamping * sum + add_const;
    }
}

static double l1_norm(const double *a, const double *b, uint64_t n) {
    double s = 0.0;
    for (uint64_t i = 0; i < n; i++) s += fabs(a[i] - b[i]);
    return s;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <prefix> [-backend ricochet|mmap] [-policy default|pin] "
            "[-warmup N] [-measure M] [-threads T] [-phys MB]\n",
            argv[0]);
        return 1;
    }
    const char *prefix = argv[1];
    int    warmup_iters  = 3;
    int    measure_iters = 1;
    int    nthreads      = omp_get_max_threads();
    size_t phys_mb       = 0;
    int    backend       = BK_RICOCHET;
    int    policy        = POL_DEFAULT;

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "-warmup")  && i + 1 < argc) warmup_iters  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-measure") && i + 1 < argc) measure_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-threads") && i + 1 < argc) nthreads      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-phys")    && i + 1 < argc) phys_mb       = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "-backend") && i + 1 < argc) {
            const char *b = argv[++i];
            if      (!strcmp(b, "ricochet")) backend = BK_RICOCHET;
            else if (!strcmp(b, "mmap"))     backend = BK_MMAP;
            else { fprintf(stderr, "unknown backend '%s'\n", b); return 1; }
        }
        else if (!strcmp(argv[i], "-policy") && i + 1 < argc) {
            const char *p = argv[++i];
            if      (!strcmp(p, "default")) policy = POL_DEFAULT;
            else if (!strcmp(p, "pin"))     policy = POL_PIN;
            else { fprintf(stderr, "unknown policy '%s'\n", p); return 1; }
        }
    }
    // The pin policy is app-managed residency, only meaningful for ricochet.
    if (backend != BK_RICOCHET) policy = POL_DEFAULT;

    char cfg_path[4096], idx_path[4096], adj_path[4096];
    snprintf(cfg_path, sizeof(cfg_path), "%s.config", prefix);
    snprintf(idx_path, sizeof(idx_path), "%s.idx",    prefix);
    snprintf(adj_path, sizeof(adj_path), "%s.adj",    prefix);

    FILE *cf = fopen(cfg_path, "r");
    if (!cf) { perror("fopen .config"); return 1; }
    uint64_t n = 0;
    if (fscanf(cf, "%" SCNu64, &n) != 1 || n == 0) {
        fprintf(stderr, "bad .config\n"); return 1;
    }
    fclose(cf);

    uint32_t *offsets = (uint32_t *)malloc(n * sizeof(uint32_t));
    if (!offsets) { perror("malloc offsets"); return 1; }
    {
        int ifd = open(idx_path, O_RDONLY);
        if (ifd < 0) { perror("open .idx"); return 1; }
        size_t total = n * sizeof(uint32_t), done = 0;
        while (done < total) {
            ssize_t r = read(ifd, (char *)offsets + done, total - done);
            if (r <= 0) { perror("read .idx"); return 1; }
            done += (size_t)r;
        }
        close(ifd);
    }

    int adj_fd = open(adj_path, O_RDONLY);
    if (adj_fd < 0) { perror("open .adj"); return 1; }
    struct stat adj_st;
    if (fstat(adj_fd, &adj_st) < 0) { perror("fstat .adj"); return 1; }
    size_t adj_size = (size_t)adj_st.st_size;
    uint64_t m      = adj_size / sizeof(uint32_t);

    printf("[pagerank] backend=%s policy=%s n=%" PRIu64 " m=%" PRIu64
           " adj=%.1f MB phys=%zu MB threads=%d\n",
           backend == BK_RICOCHET ? "ricochet" : "mmap",
           policy == POL_PIN ? "pin" : "default",
           n, m, adj_size / 1e6, phys_mb, nthreads);

    // adj[] is the backend-managed CSR neighbor array; offsets, p_curr, p_next
    // stay in ordinary memory.
    ricochet::RicochetRegion adj_region{};
    const uint32_t *adj = nullptr;
    size_t phys_pages = 0;  // ricochet cache size in pages (0 = unlimited)

    if (backend == BK_RICOCHET) {
        size_t page_size  = (size_t)sysconf(_SC_PAGESIZE);
        phys_pages = phys_mb ? (phys_mb * 1024 * 1024 / page_size) : 0;
        ricochet::cache_init(phys_pages);
        ricochet::handler_pool_init(nthreads);

        // File-backed region: fills resolve with a single MADV_POPULATE_READ and
        // eviction uses the batched MADV_DONTNEED default — no fill/evict handler.
        ricochet::Handlers handlers;
        if (ricochet::region_init(&adj_region, adj_size, handlers,
                                  /*use_uffd=*/false, adj_fd) < 0) {
            perror("region_init"); return 1;
        }
        madvise(adj_region.addr, adj_region.size, MADV_NOHUGEPAGE);
        adj = (const uint32_t *)adj_region.addr;
    } else {
        void *a = mmap(nullptr, adj_size, PROT_READ, MAP_PRIVATE, adj_fd, 0);
        if (a == MAP_FAILED) { perror("mmap adj"); return 1; }
        madvise(a, adj_size, MADV_NOHUGEPAGE);
        adj = (const uint32_t *)a;
    }

    double *p_curr = (double *)malloc(n * sizeof(double));
    double *p_next = (double *)malloc(n * sizeof(double));
    if (!p_curr || !p_next) { perror("malloc pagerank arrays"); return 1; }
    for (uint64_t i = 0; i < n; i++) { p_curr[i] = 1.0 / (double)n; p_next[i] = 0.0; }

    // --- KVM warmup (UFFD, OMP barriers safe) ---
    printf("[pagerank] warmup: %d iter(s), %d threads\n", warmup_iters, nthreads);
    for (int it = 0; it < warmup_iters; it++) {
        pagerank_iter_parallel(adj, offsets, n, m, p_curr, p_next, nthreads);
        double l1 = l1_norm(p_curr, p_next, n);
        printf("[pagerank] warmup %d  L1=%.6f\n", it, l1);
        double *tmp = p_curr; p_curr = p_next; p_next = tmp;
        for (uint64_t i = 0; i < n; i++) p_next[i] = 0.0;
    }

    printf("[pagerank] taking checkpoint\n");
    fflush(stdout);
    m5op_addr = 0xFFFF0000;
    map_m5_mem();
    m5_checkpoint_addr(0, 0);

    // --- O3 measure ---
    if (backend == BK_RICOCHET)
        ricochet::stop_handler_pool();

    // Switch the region to app-managed residency for the pin policy: cold-evict
    // it (so the measured phase starts empty and every access faults through the
    // policy) and install the on_fault hook.  Per-thread pin budget below.
    uint64_t pin_pages_per_thread = 0;
    if (policy == POL_PIN) {
        cold_evict_region(&adj_region);
        adj_region.handlers.on_fault = pin_on_fault;
        ricochet::region_set_app_managed(&adj_region, true);
        uint64_t per_thread = phys_pages / (uint64_t)nthreads;
        // Reserve one evict batch (64) + current page for streaming headroom.
        pin_pages_per_thread = per_thread > 128 ? per_thread - 65 : per_thread / 2;
        printf("[pagerank] policy=pin: pinning up to %" PRIu64 " pages/thread\n",
               pin_pages_per_thread);
    }

    for (int it = 0; it < measure_iters; it++) {
        uint64_t faults_before =
            backend == BK_RICOCHET ? ricochet::global_cache().upfFaultCount.load() : 0;

        if (backend == BK_RICOCHET) {
            // Each thread registers for UINTR, does its work, then unregisters.
            // The barrier before enable_uintr is the last OMP barrier during UIF=1.
            #pragma omp parallel num_threads(nthreads)
            {
                ricochet::region_register_thread();

                // Per-thread pin window over this worker's contiguous adj slice.
                if (policy == POL_PIN) {
                    int tid = omp_get_thread_num();
                    uint64_t v0 = (uint64_t)tid * n / (uint64_t)nthreads;
                    uint64_t v1 = (uint64_t)(tid + 1) * n / (uint64_t)nthreads;
                    uint64_t first_pg = ((uint64_t)offsets[v0] * sizeof(uint32_t)) >> 12;
                    uint64_t last_byte = (uint64_t)(v1 < n ? offsets[v1] : (uint32_t)m)
                                         * sizeof(uint32_t);
                    uint64_t slice_pages = ((last_byte + 4095) >> 12) - first_pg;
                    uint64_t pin = pin_pages_per_thread < slice_pages
                                       ? pin_pages_per_thread : slice_pages;
                    tl_pin_end_pg     = first_pg + pin;
                    tl_prev_stream_pg = ~0ULL;
                    tl_evict_n        = 0;
                }

                #pragma omp barrier              // all registered before any enables
                ricochet::region_enable_uintr(); // UIF=1 — no more barriers after this

                pagerank_iter_upf(adj, offsets, n, m, p_curr, p_next);

                ricochet::region_unregister_thread(); // UIF=0 before parallel-section exit barrier
                if (policy == POL_PIN) pin_flush(&adj_region);
            }
        } else {
            // mmap backend: the kernel serves faults; OMP barriers are safe.
            pagerank_iter_parallel(adj, offsets, n, m, p_curr, p_next, nthreads);
        }

        uint64_t faults =
            backend == BK_RICOCHET
                ? ricochet::global_cache().upfFaultCount.load() - faults_before : 0;
        double l1 = l1_norm(p_curr, p_next, n);
        printf("[pagerank] measure %d  L1=%.6f  faults=%" PRIu64 "\n", it, l1, faults);

        double *tmp = p_curr; p_curr = p_next; p_next = tmp;
        for (uint64_t i = 0; i < n; i++) p_next[i] = 0.0;
    }

    if (backend == BK_RICOCHET)
        ricochet::region_destroy(&adj_region);
    else
        munmap((void *)adj, adj_size);
    free(p_curr); free(p_next); free(offsets);
    close(adj_fd);
    return 0;
}
