// Safe PG sweep: test ONLY the working persistent kernel on ONE shape, with
// explicit sync + error checks + 2s cooldown between runs.
//
// Compile:
//   hipcc -O3 --offload-arch=gfx1100 benchmarks/vision/bench_pg_safe.cpp -o /tmp/bench_pg_safe
// Run:
//   /tmp/bench_pg_safe
//
// This file embeds the kernel directly (no dlopen) so a launch syntax bug
// would have been caught at compile time.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

// Peak specs for gfx1100 (RX 7900 XTX)
static constexpr double PEAK_TFLOPS = 122.8;

// Kernel: exact copy of kernels/src/gemm_f16_wmma_persistent.hip
// but with `persistent_groups` promoted to a runtime parameter so the same
// compiled binary can test all 5 values without recompile.
typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float    __attribute__((ext_vector_type(8)))  float8_t;

#define NB8 8

__launch_bounds__(32, 8)
__global__ void gemm_f16_wmma_persistent_runtime_pg(
    const _Float16* __restrict__ W,   // [M, K] row-major F16
    const float*    __restrict__ X,   // [N, K] row-major F32
    float*          __restrict__ Y,   // [N, M] row-major F32 (transposed)
    int M, int K, int N,
    int n_tile_groups
) {
    const int m_tile = blockIdx.x;
    const int n_tile_start = blockIdx.y;
    const int n_tile_stride = gridDim.y;
    const int tid = threadIdx.x;

    if (m_tile * 16 >= M) return;

    const int my_a_row = m_tile * 16 + (tid & 15);
    const bool a_in_bounds = (my_a_row < M);

    for (int n_tile = n_tile_start; n_tile < n_tile_groups; n_tile += n_tile_stride) {
        const int col_start = n_tile * (16 * NB8);
        if (col_start >= N) break;

        float8_t acc[NB8];
        #pragma unroll
        for (int nb = 0; nb < NB8; nb++)
            acc[nb] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (int k0 = 0; k0 < K; k0 += 16) {
            half16_t a_reg;
            if (a_in_bounds) {
                const _Float16* src = W + (long long)my_a_row * K + k0;
                #pragma unroll
                for (int j = 0; j < 16; j++)
                    a_reg[j] = (k0 + j < K) ? src[j] : (_Float16)0.0f;
            } else {
                #pragma unroll
                for (int j = 0; j < 16; j++) a_reg[j] = (_Float16)0.0f;
            }

            #pragma unroll
            for (int nb = 0; nb < NB8; nb++) {
                const int my_b_row = col_start + nb * 16 + (tid & 15);
                half16_t b_reg;
                if (my_b_row < N) {
                    const float* src = X + (long long)my_b_row * K + k0;
                    #pragma unroll
                    for (int j = 0; j < 16; j++)
                        b_reg[j] = (k0 + j < K) ? (_Float16)src[j] : (_Float16)0.0f;
                } else {
                    #pragma unroll
                    for (int j = 0; j < 16; j++) b_reg[j] = (_Float16)0.0f;
                }
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_reg, b_reg, acc[nb]);
            }
        }

        #pragma unroll
        for (int nb = 0; nb < NB8; nb++) {
            const int out_col = col_start + nb * 16 + (tid & 15);
            if (out_col < N) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    int out_row = m_tile * 16 + 2 * j + (tid >> 4);
                    if (out_row < M) {
                        Y[(long long)out_col * M + out_row] = acc[nb][j];
                    }
                }
            }
        }
    }
}

// ---- Safe error macro ----
#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(2); \
    } \
} while (0)

// ---- Cool-down helper: let the GPU idle between runs ----
static void gpu_cooldown_seconds(int secs) {
    HIP_CHECK(hipDeviceSynchronize());
    std::this_thread::sleep_for(std::chrono::seconds(secs));
    HIP_CHECK(hipDeviceSynchronize());  // re-sync after sleep
}

// ---- One run of one PG value ----
// Returns {time_us, tflops, pct} or aborts on error.
struct Result { double time_us; double tflops; double pct; };

static Result run_one_pg(
    _Float16* d_W, float* d_X, float* d_Y,
    int M, int K, int N,
    int persistent_groups,
    int warmup_iters, int timed_iters
) {
    // Grid: [m_tiles, persistent_groups]  Block: [32]
    int m_tiles = (M + 15) / 16;
    int n_tile_groups = (N + (16 * NB8) - 1) / (16 * NB8);
    dim3 grid(m_tiles, persistent_groups);
    dim3 block(32);

    printf("  [PG=%4d] grid=(%d, %d) block=(%d) | m_tiles=%d n_tile_groups=%d ... ",
           persistent_groups, grid.x, grid.y, block.x, m_tiles, n_tile_groups);
    fflush(stdout);

    // --- Warmup ---
    for (int i = 0; i < warmup_iters; i++) {
        gemm_f16_wmma_persistent_runtime_pg<<<grid, block, 0, 0>>>(
            d_W, d_X, d_Y, M, K, N, n_tile_groups);
        HIP_CHECK(hipGetLastError());                  // catches launch errors
        HIP_CHECK(hipDeviceSynchronize());             // catches runtime errors
    }

    // --- Timed ---
    HIP_CHECK(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timed_iters; i++) {
        gemm_f16_wmma_persistent_runtime_pg<<<grid, block, 0, 0>>>(
            d_W, d_X, d_Y, M, K, N, n_tile_groups);
        HIP_CHECK(hipGetLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double time_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / timed_iters;
    double flops = 2.0 * M * K * N;
    double tflops = flops / (time_us * 1e-6) / 1e12;
    double pct = tflops / PEAK_TFLOPS * 100.0;

    printf("%8.1f us  |  %.2f TFLOP/s  |  %.1f%% of peak\n", time_us, tflops, pct);

    Result r; r.time_us = time_us; r.tflops = tflops; r.pct = pct;
    return r;
}

int main() {
    // -- Test shape --
    // Start with the EASIEST shape: `proj` (1536 x 1536 x 19520)
    // Once we confirm the sweep works without crashing, run the other 3.
    int M = 1536, K = 1536, N = 19520;
    printf("=== Safe PG sweep on shape: proj (M=%d, K=%d, N=%d) ===\n", M, K, N);
    printf("Peak: %.1f TFLOP/s\n\n", PEAK_TFLOPS);

    // 5 PG values to test. Small batch; won't exhaust VRAM.
    std::vector<int> pg_values = {384, 512, 768, 1024, 1536};

    // Allocate once. Zero-initialized so an uninitialized-read bug crashes cleanly.
    _Float16* d_W; float* d_X; float* d_Y;
    HIP_CHECK(hipMalloc((void**)&d_W, (size_t)M * K * sizeof(_Float16)));
    HIP_CHECK(hipMalloc((void**)&d_X, (size_t)N * K * sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&d_Y, (size_t)N * M * sizeof(float)));
    HIP_CHECK(hipMemset(d_W, 0, (size_t)M * K * sizeof(_Float16)));
    HIP_CHECK(hipMemset(d_X, 0, (size_t)N * K * sizeof(float)));
    HIP_CHECK(hipMemset(d_Y, 0, (size_t)N * M * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());
    printf("Buffers allocated: W=%zu MB  X=%zu MB  Y=%zu MB\n\n",
           (size_t)M*K*2/1024/1024, (size_t)N*K*4/1024/1024, (size_t)N*M*4/1024/1024);

    const int warmup = 3;
    const int timed  = 5;   // very few iters for safety first
    const int cooldown_secs = 2;

    // Find best
    int best_pg = -1;
    double best_tflops = 0.0;

    for (size_t i = 0; i < pg_values.size(); i++) {
        int pg = pg_values[i];
        Result r = run_one_pg(d_W, d_X, d_Y, M, K, N, pg, warmup, timed);
        if (r.tflops > best_tflops) { best_tflops = r.tflops; best_pg = pg; }
        if (i + 1 < pg_values.size()) {
            printf("  cooldown %ds ...\n", cooldown_secs);
            gpu_cooldown_seconds(cooldown_secs);
        }
    }

    printf("\n=== BEST: PG=%d  (%.2f TFLOP/s) ===\n", best_pg, best_tflops);

    hipFree(d_W); hipFree(d_X); hipFree(d_Y);
    HIP_CHECK(hipDeviceSynchronize());
    return 0;
}
