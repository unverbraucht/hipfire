#include <hip/hip_runtime.h>

// Multi-row: 2 warps per block, each warp processes one row independently.
// Grid = ceil(M/2). No shared memory or cross-warp sync.
// 8x unrolled inner loop for ILP.
extern "C" __global__ void gemv_q8_0_wide(
    const unsigned char* __restrict__ A_q8,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const int blocks_per_row = K / 32;
    const unsigned char* row_data = A_q8 + (size_t)row * blocks_per_row * 34;

    float sum = 0.0f;

    int bi = 0;
    for (; bi + 7 < blocks_per_row; bi += 8) {
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            const unsigned char* block = row_data + (bi + u) * 34;
            float d = (float)*((const _Float16*)block);
            signed char qval = (signed char)block[2 + lane];
            sum += d * (float)qval * x[(bi + u) * 32 + lane];
        }
    }
    for (; bi < blocks_per_row; bi++) {
        const unsigned char* block = row_data + bi * 34;
        float d = (float)*((const _Float16*)block);
        signed char qval = (signed char)block[2 + lane];
        sum += d * (float)qval * x[bi * 32 + lane];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) y[row] = sum;
}
