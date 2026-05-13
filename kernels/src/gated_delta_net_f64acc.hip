#include <hip/hip_runtime.h>

// FP64-accumulating variant of gated_delta_net_f32. Same algorithm,
// same byte-compatible signature; state values held in shared memory as
// DOUBLE across the per-token loop so per-position state-update
// ULP-level error does not compound across the recurrence.
//
// Probe-only kernel — used to test whether DeltaNet recurrent state
// precision contributes to per-position drift on Qwen3.5 LinearAttention
// layers. See `docs/investigations/2026-05-12-engine-drift-floor/` for
// the run that ruled out fp32 state precision as the source.
//
// Storage at load/store boundaries stays fp32 (matches the existing
// state buffer dtype). Internal arithmetic + shared-memory tile is fp64.
// Shared memory cost: 2x (4 KB → 8 KB at TILE_ROWS=4, HD=128).
//
// Dispatched by `gated_delta_net_f32` when HIPFIRE_DELTANET_F64=1 is set.

#define HD 128
#define TILE_ROWS 4

extern "C" __launch_bounds__(32, 4)
__global__ void gated_delta_net_f64acc(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ output,
    int n_tokens,
    int n_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= n_heads) return;
    const int tid = threadIdx.x;
    const int col = tid * 4;
    const int stride = n_heads * HD;
    const int row_start = tile * TILE_ROWS;

    // FP64 LDS tile — 4 × 128 doubles = 4 KB
    __shared__ double S_tile[TILE_ROWS * HD];

    float* S_global = state + h * HD * HD + row_start * HD;
    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_tile[i] = (double)S_global[i];

    for (int t = 0; t < n_tokens; t++) {
        const float* q_t = q + t * stride + h * HD;
        const float* k_t = k + t * stride + h * HD;
        const float* v_t = v + t * stride + h * HD;
        double alpha = exp((double)gate[t * n_heads + h]);
        double beta_v = (double)beta[t * n_heads + h];

        double k0 = (double)k_t[col],     k1 = (double)k_t[col + 1];
        double k2 = (double)k_t[col + 2], k3 = (double)k_t[col + 3];
        double q0 = (double)q_t[col],     q1 = (double)q_t[col + 1];
        double q2 = (double)q_t[col + 2], q3 = (double)q_t[col + 3];

        for (int r = 0; r < TILE_ROWS; r++) {
            int row = row_start + r;
            double* sr = S_tile + r * HD + col;
            double s0 = sr[0], s1 = sr[1], s2 = sr[2], s3 = sr[3];

            double kv = s0 * k0 + s1 * k1 + s2 * k2 + s3 * k3;
            for (int o = 16; o > 0; o >>= 1)
                kv += __shfl_down(kv, o);
            kv = __shfl(kv, 0);

            double delta = ((double)v_t[row] - alpha * kv) * beta_v;

            s0 = alpha * s0 + k0 * delta;
            s1 = alpha * s1 + k1 * delta;
            s2 = alpha * s2 + k2 * delta;
            s3 = alpha * s3 + k3 * delta;
            sr[0] = s0; sr[1] = s1; sr[2] = s2; sr[3] = s3;

            double out_v = s0 * q0 + s1 * q1 + s2 * q2 + s3 * q3;
            for (int o = 16; o > 0; o >>= 1)
                out_v += __shfl_down(out_v, o);
            if (tid == 0)
                output[t * stride + h * HD + row] = (float)out_v;
        }
    }

    for (int i = tid; i < TILE_ROWS * HD; i += 32)
        S_global[i] = (float)S_tile[i];
}
