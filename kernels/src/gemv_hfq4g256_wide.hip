#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_hfq4g256_wide(
    const char* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M, int K
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    const int row = blockIdx.x * 2 + warp_id;
    if (row >= M) return;

    const int groups_per_row = K / 256;
    const int row_bytes = groups_per_row * 136;
    const char* row_ptr = A + (long long)row * row_bytes;

    float acc = 0.0f;

    for (int g = 0; g < groups_per_row; g++) {
        const char* gptr = row_ptr + g * 136;
        float scale = __builtin_bit_cast(float, *(const unsigned int*)(gptr));
        float zero  = __builtin_bit_cast(float, *(const unsigned int*)(gptr + 4));
        const unsigned char* nibbles = (const unsigned char*)(gptr + 8);

        int base_idx = g * 256 + lane * 8;
        int byte_off = lane * 4;

        unsigned char b0 = nibbles[byte_off];
        unsigned char b1 = nibbles[byte_off + 1];
        unsigned char b2 = nibbles[byte_off + 2];
        unsigned char b3 = nibbles[byte_off + 3];

        acc += (scale * (float)(b0 & 0xF) + zero) * x[base_idx]
             + (scale * (float)(b0 >> 4)  + zero) * x[base_idx + 1]
             + (scale * (float)(b1 & 0xF) + zero) * x[base_idx + 2]
             + (scale * (float)(b1 >> 4)  + zero) * x[base_idx + 3]
             + (scale * (float)(b2 & 0xF) + zero) * x[base_idx + 4]
             + (scale * (float)(b2 >> 4)  + zero) * x[base_idx + 5]
             + (scale * (float)(b3 & 0xF) + zero) * x[base_idx + 6]
             + (scale * (float)(b3 >> 4)  + zero) * x[base_idx + 7];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    if (lane == 0) y[row] = acc;
}
