// PageRank benchmark with ricochet-managed graph adjacency data for gem5.
//
// Expects the ligra binary graph format (produced by adjToBinary):
//   <prefix>.config  — text: n (vertex count)
//   <prefix>.idx     — n uint32_t CSR offsets
//   <prefix>.adj     — m uint32_t neighbor IDs  [ricochet-managed]
//
// Flow:
//   KVM:  run -warmup iterations (UFFD serves page faults), then checkpoint
//   O3:   each OMP thread registers for UINTR, runs -measure iterations,
//         then unregisters before any cross-thread synchronisation.
//
// Usage: pagerank_ricochet <prefix> [-warmup N] [-measure M] [-threads T] [-phys MB]

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

struct GraphCtx {
    int   fd;
    void *base;
};

static void graph_fill(void *buf, size_t offset, void *ctx) {
    auto *g = static_cast<GraphCtx *>(ctx);
    pread(g->fd, buf, 4096, static_cast<off_t>(offset));
}

static void graph_evict(size_t offset, void *ctx) {
    auto *g = static_cast<GraphCtx *>(ctx);
    madvise(static_cast<char *>(g->base) + offset, 4096, MADV_DONTNEED);
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

    for (uint64_t v = my_start; v < my_end; v++) {
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
            "Usage: %s <prefix> [-warmup N] [-measure M] [-threads T] [-phys MB]\n",
            argv[0]);
        return 1;
    }
    const char *prefix = argv[1];
    int    warmup_iters  = 3;
    int    measure_iters = 1;
    int    nthreads      = omp_get_max_threads();
    size_t phys_mb       = 0;

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "-warmup")  && i + 1 < argc) warmup_iters  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-measure") && i + 1 < argc) measure_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-threads") && i + 1 < argc) nthreads      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-phys")    && i + 1 < argc) phys_mb       = (size_t)atoll(argv[++i]);
    }

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

    printf("[pagerank] n=%" PRIu64 " m=%" PRIu64 " adj=%.1f MB phys=%zu MB threads=%d\n",
           n, m, adj_size / 1e6, phys_mb, nthreads);

    size_t page_size  = (size_t)sysconf(_SC_PAGESIZE);
    size_t phys_pages = phys_mb ? (phys_mb * 1024 * 1024 / page_size) : 0;
    ricochet::cache_init(phys_pages);
    ricochet::handler_pool_init(nthreads);

    GraphCtx gctx = {adj_fd, nullptr};
    ricochet::Handlers handlers;
    handlers.fill  = graph_fill;
    handlers.evict = graph_evict;
    handlers.ctx   = &gctx;

    ricochet::RicochetRegion adj_region{};
    if (ricochet::region_init(&adj_region, adj_size, handlers, /*use_uffd=*/false) < 0) {
        perror("region_init"); return 1;
    }
    gctx.base = adj_region.addr;
    madvise(adj_region.addr, adj_region.size, MADV_NOHUGEPAGE);

    const uint32_t *adj = (const uint32_t *)adj_region.addr;

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

    // --- O3 measure (UPF, no OMP barriers while UIF=1) ---
    ricochet::stop_handler_pool();

    for (int it = 0; it < measure_iters; it++) {
        uint64_t faults_before = ricochet::global_cache().evictedPageCount.load();

        // Each thread registers for UINTR, does its work, then unregisters.
        // The barrier before enable_uintr is the last OMP barrier during UIF=1.
        #pragma omp parallel num_threads(nthreads)
        {
            ricochet::region_register_thread();
            #pragma omp barrier              // all registered before any enables
            ricochet::region_enable_uintr(); // UIF=1 — no more barriers after this

            pagerank_iter_upf(adj, offsets, n, m, p_curr, p_next);

            ricochet::region_unregister_thread(); // UIF=0 before parallel-section exit barrier
        }

        uint64_t faults = ricochet::global_cache().evictedPageCount.load() - faults_before;
        double l1 = l1_norm(p_curr, p_next, n);
        printf("[pagerank] measure %d  L1=%.6f  faults=%" PRIu64 "\n", it, l1, faults);

        double *tmp = p_curr; p_curr = p_next; p_next = tmp;
        for (uint64_t i = 0; i < n; i++) p_next[i] = 0.0;
    }

    ricochet::region_destroy(&adj_region);
    free(p_curr); free(p_next); free(offsets);
    close(adj_fd);
    return 0;
}
