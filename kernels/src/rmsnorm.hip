#include <hip/hip_runtime.h>

extern "C" __global__ void rmsnorm_f32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n, float eps
) {
    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[blockIdx.x * n + i];
        sum_sq += v * v;
    }
    sdata[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float rms = rsqrtf(sdata[0] / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int idx = blockIdx.x * n + i;
        out[idx] = x[idx] * weight[i] * rms;
    }
}

// Probe-only fp64-accumulating RMSNorm. Same on-disk weights and outputs
// (still f32), but the sum-of-squares and rms are computed in double to
// eliminate accumulation-precision as a candidate for the engine-drift
// floor on Qwen3.5 FullAttention layers. Wire in via a HIPFIRE_RMSNORM_F64
// env-gated dispatch if this kernel produces a different forward.
extern "C" __global__ void rmsnorm_f32_f64acc(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int n, float eps
) {
    extern __shared__ double sdata_d[];
    double sum_sq = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double v = (double)x[blockIdx.x * n + i];
        sum_sq += v * v;
    }
    sdata_d[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata_d[threadIdx.x] += sdata_d[threadIdx.x + s];
        __syncthreads();
    }
    double rms = 1.0 / sqrt(sdata_d[0] / (double)n + (double)eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        int idx = blockIdx.x * n + i;
        out[idx] = (float)((double)x[idx] * (double)weight[i] * rms);
    }
}
