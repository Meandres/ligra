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
// Warmup runs in KVM (fast) to establish residency, then the checkpoint switches
// to O3 and the timed phase runs the SAME window of W source vertices M times
// (-measure M -verts W), timing every repetition separately.  Repeating a fixed
// window measures steady-state throughput on a stable working set: its reused
// high-degree neighbour pages stay hot under a large cache but get evicted and
// re-fetched under a small one — that is how cache size and the replacement
// policy show up in per-repetition throughput.  The window starts at
// -voff (default n/2) so no source vertex is a mega-hub — in RMAT a low-ID vertex
// has huge degree and processing it as a source scans most of the graph.
// Throughput is reported per range as edges/kcycle (degree-skew-normalised).
//
// Usage: triangle_ricochet <prefix> [-backend ricochet|mmap]
//        [-policy default|degree] [-verts W] [-warmup Wu] [-voff V] [-measure M]
//        [-threads T] [-phys MB]

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

// Per-thread FIFO slots for non-pinned (streaming) pages: a small fixed
// reserve so nearly the whole budget goes to the static pin set.
static uint64_t g_ring = 512;

// Degree-aware pin set (read-only during the measured phase): a bitmap over the
// region's pages; a set bit means "high-degree, keep resident".
static const uint8_t *g_pin_bitmap = nullptr;
static uint64_t       g_pin_pages  = 0;

// Per-thread FIFO of the last g_ring non-pinned pages; older ones are evicted.
static thread_local uint64_t *tl_ring      = nullptr;
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
        tl_ring = (uint64_t *)malloc(g_ring * sizeof(uint64_t));
        for (uint64_t i = 0; i < g_ring; i++) tl_ring[i] = ~0ULL;
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
    tl_ring_pos = (tl_ring_pos + 1) % (int)g_ring;
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

// Heartbeat print usable inside the UIF=1 measured region (single write(),
// no stdio locks, short line so concurrent per-thread writes don't tear).
static inline void progress(int tid, uint64_t done, uint64_t total) {
    char b[80];
    int l = snprintf(b, sizeof b, "[triangle]   t%d %llu/%llu\n", tid,
                     (unsigned long long)done, (unsigned long long)total);
    (void)!write(1, b, l);
}

// Heartbeat stride (source vertices) in the measured window — kept small so
// progress is visible under very slow O3; override with TR_HB=<n>.
static uint64_t hb_step() {
    const char *e = getenv("TR_HB");
    uint64_t s = e ? strtoull(e, nullptr, 10) : 10;
    return s ? s : 1;
}

static inline uint64_t rdtsc() {
    unsigned lo, hi;
    __asm__ __volatile__("lfence; rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

// Split resident memory into anonymous (temp scratch: offsets, stacks, sort
// vector, OMP/libc) vs file-backed (the mmap'd adj cache).  RssAnon is what a
// cgroup limit must accommodate ON TOP of the intended adj cache budget so the
// mmap backend isn't starved of adj cache by its own scratch data.
static void report_rss(const char *tag) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return;
    char line[256];
    long vmrss = 0, anon = 0, file = 0;
    while (fgets(line, sizeof line, f)) {
        sscanf(line, "VmRSS: %ld kB", &vmrss);
        sscanf(line, "RssAnon: %ld kB", &anon);
        sscanf(line, "RssFile: %ld kB", &file);
    }
    fclose(f);
    printf("[mem] %s  VmRSS=%ld MB  RssAnon(temp)=%ld MB  RssFile(adj)=%ld MB\n",
           tag, vmrss / 1024, anon / 1024, file / 1024);
    fflush(stdout);
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
                               uint64_t n, uint64_t m, uint64_t s, uint64_t e,
                               int tid = -1, uint64_t prog_step = 0) {
    uint64_t tri = 0;
    for (uint64_t u = s; u < e; u++) {
        if (prog_step && tid >= 0 && (u - s) % prog_step == 0)
            progress(tid, u - s, e - s);
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

// Process the source-vertex window [s, e) once across nthreads.  For the
// ricochet backend each worker registers + enables UINTR so faults are handled
// through the app-managed residency path (as in the measured region); the join
// at the end of the parallel region happens with UIF=0 (region_unregister_thread
// clears it), so it is safe to time or barrier around this call.  When verbose,
// each worker emits heartbeats over its slice.
static uint64_t run_window(bool ricochet, int policy, ricochet::RicochetRegion *rr,
                           const uint32_t *adj, const uint32_t *offsets,
                           uint64_t n, uint64_t m, uint64_t s, uint64_t e,
                           int nthreads, bool verbose) {
    uint64_t parts[MAXT] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        if (ricochet) {
            ricochet::region_register_thread();
            // Degree FIFO persists across the sweep's ranges (self-inits on the
            // first fault): resetting it per range would make each range forget —
            // and thus stop evicting — the pages it tracked, so residency would
            // grow past the cache budget instead of streaming.
            #pragma omp barrier
            ricochet::region_enable_uintr();
        }
        uint64_t span = e - s;
        uint64_t ws = s + (uint64_t)tid * span / (uint64_t)nthreads;
        uint64_t we = s + (uint64_t)(tid + 1) * span / (uint64_t)nthreads;
        parts[tid] = triangle_range(adj, offsets, n, m, ws, we,
                                    verbose ? tid : -1, verbose ? hb_step() : 0);
        if (ricochet) {
            ricochet::region_unregister_thread();
            if (policy == POL_DEGREE) degree_flush(rr);
        }
    }
    uint64_t tri = 0;
    for (int t = 0; t < nthreads; t++) tri += parts[t];
    return tri;
}

// KVM warmup (no UINTR): plain-read the adjacency of source vertices [s, e) and
// touch each neighbour's adjacency, faulting the working set (hub pages) in from
// the backing file.  Cheap in KVM; establishes residency before the checkpoint.
static void kvm_warm_pass(const uint32_t *adj, const uint32_t *offsets,
                          uint64_t n, uint64_t m, uint64_t s, uint64_t e, int T) {
    uint64_t sink = 0;
    #pragma omp parallel for num_threads(T) schedule(dynamic, 64) reduction(+:sink)
    for (uint64_t u = s; u < e; u++) {
        uint32_t us = offsets[u];
        uint32_t ue = (u + 1 < n) ? offsets[u + 1] : (uint32_t)m;
        for (uint32_t k = us; k < ue; k++) {
            uint32_t v = adj[k];                       // read source adjacency
            sink += adj[offsets[v]];                   // touch neighbour's page
        }
    }
    if (sink == ~0ULL) fprintf(stderr, "");            // keep the reads live
}

// MADV_POPULATE_READ every contiguous run of pinned pages so the degree policy's
// pinned set is resident before measurement (rest of the region stays cold).
static void populate_pinned(void *base) {
    uint64_t p = 0;
    while (p < g_pin_pages) {
        if (!is_pinned(p)) { p++; continue; }
        uint64_t q = p;
        while (q < g_pin_pages && is_pinned(q)) q++;
        madvise((char *)base + p * 4096, (size_t)(q - p) * 4096, MADV_POPULATE_READ);
        p = q;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <prefix> [-backend ricochet|mmap] [-policy default|degree] "
            "[-verts W] [-warmup Wu] [-measure M] [-threads T] [-phys MB]\n",
            argv[0]);
        return 1;
    }
    const char *prefix = argv[1];
    uint64_t warmup_verts  = 4096;   // source vertices warmed (untimed) to fill residency
    int    measure_iters = 1;        // consecutive source ranges swept & timed (see the loop)
    int    nthreads      = omp_get_max_threads();
    size_t phys_mb       = 0;
    int    backend       = BK_RICOCHET;
    int    policy        = POL_DEFAULT;
    uint64_t verts       = 2048;   // source vertices in the timed window
    uint64_t voff        = ~0ULL;  // window start; default (sentinel) = n/2, past the hubs

    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "-warmup")  && i + 1 < argc) warmup_verts  = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "-voff")    && i + 1 < argc) voff          = (uint64_t)atoll(argv[++i]);
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
           " adj=%.1f MB phys=%zu MB threads=%d warmup=%" PRIu64 " verts=%" PRIu64 "\n",
           backend == BK_RICOCHET ? "ricochet" : "mmap",
           policy == POL_DEGREE ? "degree" : "default",
           n, m, adj_size / 1e6, phys_mb, nthreads, warmup_verts, verts);

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
        // Disable readahead so every backend fetches one page per fault;
        // otherwise mmap's cluster reads amortize the device round trip in a
        // way no userspace fault path is allowed to.
        madvise(a, adj_size, MADV_RANDOM);
        adj = (const uint32_t *)a;
    }

    bool ric = (backend == BK_RICOCHET);

    // Window layout: warm [warm_beg, warm_end) then time [meas_beg, meas_end).
    // Default start is n/2 — deep in the low-degree tail — so neither warmup nor
    // the timed window processes a mega-hub *as a source* (that alone would make
    // one source scan most of the graph).  Hub adjacencies are still read heavily
    // as *neighbours*, so the reuse the degree policy exploits is preserved.
    if (voff == ~0ULL) voff = n / 2;
    if (voff > n) voff = 0;
    uint64_t warm_beg = voff;
    uint64_t warm_end = warm_beg + warmup_verts;                 if (warm_end > n) warm_end = n;
    uint64_t meas_beg = warm_end;                                if (meas_beg > n) meas_beg = n;
    uint64_t meas_end = meas_beg + (uint64_t)measure_iters * verts;  // end of the whole sweep
    if (meas_end > n) meas_end = n;

    // --- Policy setup (KVM, before the checkpoint) ---
    if (policy == POL_DEGREE) {
        // Pin pages by access weight, not by vertex degree.  A page holds many
        // small adjacency lists (or part of one large list), and in triangle
        // counting a vertex's list is scanned ~degree(v) times, so the reuse of a
        // *page* is the sum of degree(v) over every vertex whose list touches it.
        // We rank pages by that weight and pin the heaviest up to the cache budget
        // (leaving a residency-proportional streaming reserve for the FIFOs).
        //
        // This is a single linear pass over offsets[] (the CSR degree metadata),
        // not a traversal of the adjacency edges: it never reads the region.  The
        // pass must complete before ranking because top-budget selection is global.
        uint64_t total_pages = (adj_size + 4095) / 4096;

        // Each vertex adds its full degree to every page its list spans, since
        // each scan of the list reads all of those pages.
        std::vector<uint64_t> weight(total_pages, 0);
        for (uint64_t v = 0; v < n; v++) {
            uint32_t e   = (v + 1 < n) ? offsets[v + 1] : (uint32_t)m;
            uint32_t deg = e - offsets[v];
            if (deg == 0) continue;
            uint64_t b0 = (uint64_t)offsets[v] * sizeof(uint32_t);
            uint64_t b1 = (uint64_t)e * sizeof(uint32_t);
            for (uint64_t p = b0 >> 12; p <= (b1 - 1) >> 12; p++)
                weight[p] += deg;
        }

        // Rank pages by descending weight.
        std::vector<uint32_t> order(total_pages);
        for (uint64_t p = 0; p < total_pages; p++) order[p] = (uint32_t)p;
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) { return weight[a] > weight[b]; });

        // Static pin/stream split: a fixed 512-slot per-thread streaming FIFO;
        // every remaining budget page is pinned by weight.  When the budget is
        // smaller than the reserve, pin half of it and let the FIFOs stream.
        g_ring = 512;
        uint64_t stream = (uint64_t)nthreads * g_ring;
        uint64_t budget = phys_pages > stream ? phys_pages - stream : phys_pages / 2;
        if (budget > total_pages) budget = total_pages;

        // Pin the top-budget pages that carry any reuse.
        uint8_t *bitmap = (uint8_t *)calloc((total_pages + 7) / 8, 1);
        uint64_t pinned = 0;
        for (uint64_t i = 0; i < total_pages && pinned < budget; i++) {
            uint32_t p = order[i];
            if (weight[p] == 0) break;
            bitmap[p >> 3] |= (uint8_t)(1u << (p & 7));
            pinned++;
        }
        g_pin_bitmap = bitmap;
        g_pin_pages  = total_pages;
        adj_region.handlers.on_fault = degree_on_fault;
        ricochet::region_set_app_managed(&adj_region, true);
        printf("[triangle] policy=degree: pinned %" PRIu64 " / %" PRIu64 " pages"
               " (stream reserve %" PRIu64 " pages, ring %" PRIu64 "/thread)\n",
               pinned, total_pages, stream, g_ring);
    }

    // --- Warmup (KVM, before the checkpoint) ---
    // Establish the residency the timed window will start from, cheaply in KVM.
    // degree : pinned pages resident (populated), everything else cold so the
    //          streaming (non-pinned) reads fault via UPF in O3 = the overhead
    //          we measure.  default/mmap : fault the warmup window's working set
    //          (source + hub-neighbour pages) in from the backing file.
    if (policy == POL_DEGREE) {
        cold_evict_region(&adj_region);
        populate_pinned(adj_region.addr);
    } else {
        kvm_warm_pass(adj, offsets, n, m, warm_beg, warm_end, nthreads);
    }
    // The handler pool is deliberately kept alive through the measured phase.
    // A fault that traps to the kernel on a UPF-stamped page (instead of being
    // hardware-delivered) queues on the region's uffd; with the pool stopped
    // that thread sleeps forever -- the intermittent whole-run hang we observed.
    // With the pool alive, such a misroute costs one slow (uffd-path) fault and
    // shows up in the uffd= counter of the measure lines.

    printf("[triangle] warmup [%" PRIu64 ",%" PRIu64 ")  measure [%" PRIu64 ",%" PRIu64 ")\n",
           warm_beg, warm_end, meas_beg, meas_end);
    report_rss("pre-checkpoint");
    printf("[triangle] taking checkpoint\n");
    fflush(stdout);
    m5op_addr = 0xFFFF0000;
    map_m5_mem();
    m5_checkpoint_addr(0, 0);

    // --- O3: timed measured phase — a forward SWEEP of consecutive source ranges.
    // Each of the `measure_iters` ranges is `verts` source vertices, timed on its
    // own.  Warmup made the range just before meas_beg resident, so the sweep starts
    // warm and moves forward: the shared high-degree *neighbour* pages are reused
    // across ranges, so a large cache keeps them resident (throughput stays high)
    // while a small cache evicts and re-fetches them (per-range faults rise and
    // throughput drops) — that's the cache-size / replacement-policy sensitivity we
    // want to expose (re-running one resident window, as before, showed none).
    if (verts == 0) measure_iters = 0;
    else if (meas_beg + (uint64_t)measure_iters * verts > n)
        measure_iters = (int)((n > meas_beg ? n - meas_beg : 0) / verts);

    // Accumulate the measured lines so we can hand them to the host via
    // m5 writefile (result_benchmark.txt) for the bench_runner to parse — the
    // same mechanism db_bench uses.
    std::string result_log;
    {
        char hdr[256];
        snprintf(hdr, sizeof hdr,
                 "[triangle] backend=%s policy=%s ranges=%d range_verts=%" PRIu64
                 " cache=%zu MB threads=%d\n",
                 ric ? "ricochet" : "mmap", policy == POL_DEGREE ? "degree" : "default",
                 measure_iters, verts, phys_mb, nthreads);
        result_log += hdr;
    }

    for (int it = 0; it < measure_iters; it++) {
        uint64_t rbeg = meas_beg;
        uint64_t rend = rbeg + verts; if (rend > n) rend = n;
        uint64_t e_beg = offsets[rbeg];
        uint64_t e_end = (rend < n) ? offsets[rend] : m;
        uint64_t win_edges = e_end - e_beg;   // edges in THIS range (throughput basis)

        uint64_t faults_before = ric ? ricochet::global_cache().upfFaultCount.load() : 0;
        uint64_t uffd_before   = ric ? ricochet::uffd_handled_count() : 0;
        uint64_t c0 = rdtsc();
        uint64_t tri = run_window(ric, policy, &adj_region, adj, offsets, n, m,
                                  rbeg, rend, nthreads, /*verbose=*/true);
        uint64_t c1 = rdtsc();
        uint64_t faults = ric ? ricochet::global_cache().upfFaultCount.load() - faults_before : 0;
        uint64_t uffd   = ric ? ricochet::uffd_handled_count() - uffd_before : 0;
        uint64_t cyc = c1 - c0;
        double eppk = cyc ? (double)win_edges / (double)cyc * 1000.0 : 0.0;
        char line[256];
        snprintf(line, sizeof line,
                 "[triangle] measure %d  triangles=%" PRIu64 "  edges=%" PRIu64
                 "  faults=%" PRIu64 "  cycles=%" PRIu64 "  edges_per_kcycle=%.3f"
                 "  uffd=%" PRIu64 "\n",
                 it, tri, win_edges, faults, cyc, eppk, uffd);
        fputs(line, stdout);
        fflush(stdout);
        result_log += line;
    }

    // Hand results to the host and end the simulation (no-op cleanup follows in
    // case m5_exit is inert in some build).
    m5_write_file_addr((void *)result_log.c_str(), (uint64_t)result_log.size(),
                       0, "result_benchmark.txt");
    m5_exit_addr(0);

    if (backend == BK_RICOCHET)
        ricochet::region_destroy(&adj_region);
    else
        munmap((void *)adj, adj_size);
    free(offsets);
    close(adj_fd);
    return 0;
}
