// Triangle-counting benchmark with ricochet-managed graph adjacency for gem5.
//
// Reads the ligra binary graph format (produced by adjToBinary):
//   <prefix>.config  — text: n (vertex count)
//   <prefix>.idx     — n uint32_t CSR offsets
//   <prefix>.adj     — m uint32_t neighbor IDs  [backend-managed]  (SORTED)
//
// Why triangle counting: for each edge (u,v) it intersects adj(u) and adj(v),
// so a high-degree vertex's adjacency is read ~degree times — access frequency
// is strongly skewed by degree.  That makes it the ideal workload to show a
// degree-aware replacement policy beating an oblivious one (unlike PageRank,
// whose adjacency access is a uniform sequential stream).
//
// Policy (ricochet backend, app-managed residency):
//   default — ricochet's built-in S3-FIFO cache.
//   degree  — pin the adjacency pages of the highest-degree vertices up to the
//             cache budget (that's where the reuse is); the rest run through a
//             small per-thread FIFO.  ricochet only fills; the app evicts.
//
// Usage: triangle_ricochet <prefix> [-backend ricochet|mmap]
//        [-policy default|degree] [-verts W] [-warmup N] [-measure M]
//        [-threads T] [-phys MB]

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
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

enum Backend { BK_RICOCHET, BK_MMAP };
enum Policy  { POL_DEFAULT, POL_DEGREE };

static const int   MAXT = 256;
static const int   RING = 512;   // per-thread FIFO slots for non-pinned pages

// Degree-aware pin set (read-only during the measured phase): a bitmap over the
// region's pages; a set bit means "high-degree, keep resident".
static const uint8_t *g_pin_bitmap = nullptr;
static uint64_t       g_pin_pages  = 0;

// Per-thread FIFO of the last RING non-pinned pages; older ones are evicted.
static thread_local uint64_t tl_ring[RING];
static thread_local int      tl_ring_pos  = 0;
static thread_local bool     tl_ring_init = false;
static thread_local size_t   tl_evict_buf[64];
static thread_local int      tl_evict_n   = 0;

static inline bool is_pinned(uint64_t pg) {
    return g_pin_bitmap && pg < g_pin_pages &&
           ((g_pin_bitmap[pg >> 3] >> (pg & 7)) & 1u);
}

// ricochet calls this after filling a faulted page (app-managed region).
static void degree_on_fault(ricochet::RicochetRegion *r, size_t offset, void *) {
    uint64_t pg = offset >> 12;
    if (is_pinned(pg)) return;                     // degree-hot: keep resident
    if (!tl_ring_init) {
        for (int i = 0; i < RING; i++) tl_ring[i] = ~0ULL;
        tl_ring_init = true;
    }
    uint64_t old = tl_ring[tl_ring_pos];
    if (old != ~0ULL) {
        tl_evict_buf[tl_evict_n++] = (size_t)(old << 12);
        if (tl_evict_n == 64) {
            ricochet::do_evict_pages(r, tl_evict_buf, tl_evict_n);
            tl_evict_n = 0;
        }
    }
    tl_ring[tl_ring_pos] = pg;
    tl_ring_pos = (tl_ring_pos + 1) % RING;
}

static void degree_flush(ricochet::RicochetRegion *r) {
    if (tl_evict_n) { ricochet::do_evict_pages(r, tl_evict_buf, tl_evict_n); tl_evict_n = 0; }
}

// Drop every PTE so the app-managed measured phase starts from empty residency.
static void cold_evict_region(ricochet::RicochetRegion *r) {
    size_t npages = r->size / 4096;
    size_t buf[512];
    for (size_t base = 0; base < npages; ) {
        int k = 0;
        while (k < 512 && base < npages) buf[k++] = (base++) * 4096;
        ricochet::do_evict_pages(r, buf, k);
    }
}

// Count common neighbors w > v of two sorted adjacency lists (triangle u<v<w).
static inline uint64_t intersect_gt(const uint32_t *au, uint32_t du,
                                    const uint32_t *av, uint32_t dv, uint32_t v) {
    uint64_t c = 0; uint32_t i = 0, j = 0;
    while (i < du && j < dv) {
        uint32_t a = au[i], b = av[j];
        if (a <= v) { i++; continue; }
        if (b <= v) { j++; continue; }
        if      (a == b) { c++; i++; j++; }
        else if (a <  b) { i++; }
        else             { j++; }
    }
    return c;
}

// Process source vertices [s, e): for each edge (u,v) with v>u, count common
// neighbors.  adj[] is ricochet-managed; offsets is in ordinary memory.
static uint64_t triangle_range(const uint32_t *adj, const uint32_t *offsets,
                               uint64_t n, uint64_t m, uint64_t s, uint64_t e) {
    uint64_t tri = 0;
    for (uint64_t u = s; u < e; u++) {
        uint32_t us = offsets[u];
        uint32_t ue = (u + 1 < n) ? offsets[u + 1] : (uint32_t)m;
        const uint32_t *au = adj + us;
        uint32_t du = ue - us;
        for (uint32_t k = 0; k < du; k++) {
            uint32_t v = au[k];
            if (v <= u) continue;
            uint32_t vs = offsets[v];
            uint32_t ve = (v + 1 < n) ? offsets[v + 1] : (uint32_t)m;
            tri += intersect_gt(au, du, adj + vs, ve - vs, v);
        }
    }
    return tri;
}

// OMP-parallel version for KVM warmup (barriers safe, no UINTR).
static uint64_t triangle_parallel(const uint32_t *adj, const uint32_t *offsets,
                                  uint64_t n, uint64_t m, uint64_t W, int T) {
    uint64_t total = 0;
    #pragma omp parallel for num_threads(T) schedule(dynamic, 32) reduction(+:total)
    for (uint64_t u = 0; u < W; u++)
        total += triangle_range(adj, offsets, n, m, u, u + 1);
    return total;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <prefix> [-backend ricochet|mmap] [-policy default|degree] "
            "[-verts W] [-warmup N] [-measure M] [-threads T] [-phys MB]\n",
            argv[0]);
        return 1;
    }
    const char *prefix = argv[1];
    int    warmup_iters  = 1;
    int    measure_iters = 1;
    int    nthreads      = omp_get_max_threads();
    size_t phys_mb       = 0;
    int    backend       = BK_RICOCHET;
    int    policy        = POL_DEFAULT;
    uint64_t verts       = 2048;   // source vertices processed per measured run

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "-warmup")  && i + 1 < argc) warmup_iters  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-measure") && i + 1 < argc) measure_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-threads") && i + 1 < argc) nthreads      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-phys")    && i + 1 < argc) phys_mb       = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "-verts")   && i + 1 < argc) verts         = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "-backend") && i + 1 < argc) {
            const char *b = argv[++i];
            if      (!strcmp(b, "ricochet")) backend = BK_RICOCHET;
            else if (!strcmp(b, "mmap"))     backend = BK_MMAP;
            else { fprintf(stderr, "unknown backend '%s'\n", b); return 1; }
        }
        else if (!strcmp(argv[i], "-policy") && i + 1 < argc) {
            const char *p = argv[++i];
            if      (!strcmp(p, "default")) policy = POL_DEFAULT;
            else if (!strcmp(p, "degree"))  policy = POL_DEGREE;
            else { fprintf(stderr, "unknown policy '%s'\n", p); return 1; }
        }
    }
    if (backend != BK_RICOCHET) policy = POL_DEFAULT;
    if (nthreads > MAXT) nthreads = MAXT;

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
    if (verts > n) verts = n;

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

    printf("[triangle] backend=%s policy=%s n=%" PRIu64 " m=%" PRIu64
           " adj=%.1f MB phys=%zu MB threads=%d verts=%" PRIu64 "\n",
           backend == BK_RICOCHET ? "ricochet" : "mmap",
           policy == POL_DEGREE ? "degree" : "default",
           n, m, adj_size / 1e6, phys_mb, nthreads, verts);

    ricochet::RicochetRegion adj_region{};
    const uint32_t *adj = nullptr;
    size_t phys_pages = 0;

    if (backend == BK_RICOCHET) {
        size_t page_size = (size_t)sysconf(_SC_PAGESIZE);
        phys_pages = phys_mb ? (phys_mb * 1024 * 1024 / page_size) : 0;
        ricochet::cache_init(phys_pages);
        ricochet::handler_pool_init(nthreads);
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

    // --- KVM warmup (UFFD, OMP barriers safe) ---
    printf("[triangle] warmup: %d iter(s)\n", warmup_iters);
    for (int it = 0; it < warmup_iters; it++) {
        uint64_t t = triangle_parallel(adj, offsets, n, m, verts, nthreads);
        printf("[triangle] warmup %d  triangles=%" PRIu64 "\n", it, t);
    }

    printf("[triangle] taking checkpoint\n");
    fflush(stdout);
    m5op_addr = 0xFFFF0000;
    map_m5_mem();
    m5_checkpoint_addr(0, 0);

    // --- O3 measure ---
    if (backend == BK_RICOCHET)
        ricochet::stop_handler_pool();

    if (policy == POL_DEGREE) {
        // Build the degree-ordered pin set: pin the adjacency pages of the
        // highest-degree vertices up to the cache budget (leave nthreads*RING
        // pages of streaming headroom for the per-thread FIFOs).
        cold_evict_region(&adj_region);
        std::vector<uint32_t> order(n);
        for (uint64_t i = 0; i < n; i++) order[i] = (uint32_t)i;
        auto deg = [&](uint32_t v) {
            uint32_t e = (v + 1 < n) ? offsets[v + 1] : (uint32_t)m;
            return e - offsets[v];
        };
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) { return deg(a) > deg(b); });

        uint64_t total_pages = (adj_size + 4095) / 4096;
        uint64_t stream = (uint64_t)nthreads * RING;
        uint64_t budget = phys_pages > stream ? phys_pages - stream : phys_pages / 2;
        if (budget > total_pages) budget = total_pages;

        uint8_t *bitmap = (uint8_t *)calloc((total_pages + 7) / 8, 1);
        uint64_t pinned = 0;
        for (uint32_t v : order) {
            if (pinned >= budget) break;
            uint64_t b0 = (uint64_t)offsets[v] * sizeof(uint32_t);
            uint64_t b1 = (uint64_t)((v + 1 < n) ? offsets[v + 1] : (uint32_t)m)
                          * sizeof(uint32_t);
            if (b1 <= b0) continue;
            for (uint64_t p = b0 >> 12; p <= (b1 - 1) >> 12 && pinned < budget; p++)
                if (!((bitmap[p >> 3] >> (p & 7)) & 1u)) {
                    bitmap[p >> 3] |= (uint8_t)(1u << (p & 7));
                    pinned++;
                }
        }
        g_pin_bitmap = bitmap;
        g_pin_pages  = total_pages;
        adj_region.handlers.on_fault = degree_on_fault;
        ricochet::region_set_app_managed(&adj_region, true);
        printf("[triangle] policy=degree: pinned %" PRIu64 " / %" PRIu64 " pages\n",
               pinned, total_pages);
    }

    for (int it = 0; it < measure_iters; it++) {
        uint64_t faults_before =
            backend == BK_RICOCHET ? ricochet::global_cache().upfFaultCount.load() : 0;
        uint64_t tri = 0;

        if (backend == BK_RICOCHET) {
            uint64_t parts[MAXT] = {0};
            #pragma omp parallel num_threads(nthreads)
            {
                ricochet::region_register_thread();
                if (policy == POL_DEGREE) { tl_ring_init = false; tl_ring_pos = 0; tl_evict_n = 0; }
                #pragma omp barrier
                ricochet::region_enable_uintr();

                int tid = omp_get_thread_num();
                uint64_t s = (uint64_t)tid * verts / (uint64_t)nthreads;
                uint64_t e = (uint64_t)(tid + 1) * verts / (uint64_t)nthreads;
                parts[tid] = triangle_range(adj, offsets, n, m, s, e);

                ricochet::region_unregister_thread();
                if (policy == POL_DEGREE) degree_flush(&adj_region);
            }
            for (int t = 0; t < nthreads; t++) tri += parts[t];
        } else {
            tri = triangle_parallel(adj, offsets, n, m, verts, nthreads);
        }

        uint64_t faults =
            backend == BK_RICOCHET
                ? ricochet::global_cache().upfFaultCount.load() - faults_before : 0;
        printf("[triangle] measure %d  triangles=%" PRIu64 "  faults=%" PRIu64 "\n",
               it, tri, faults);
    }

    if (backend == BK_RICOCHET)
        ricochet::region_destroy(&adj_region);
    else
        munmap((void *)adj, adj_size);
    free(offsets);
    close(adj_fd);
    return 0;
}
