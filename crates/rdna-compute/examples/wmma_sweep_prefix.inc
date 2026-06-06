// SPDX-License-Identifier: Apache-2.0
// Auto-generated WMMA sweep kernel prefix

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

__launch_bounds__(32, 2)
extern "C" __global__ void gemm_gate_up_mq4g256_lloyd_wmma(
    const char*    __restrict__ A_gate,
    const char*    __restrict__ A_up,
    const _Float16* __restrict__ X,
    float*         __restrict__ Y_gate,
    float*         __restrict__ Y_up,
    int gate_m, int up_m,
    int K, int N
) {
    const int total_m = gate_m + up_m;
    const int row_start = blockIdx.x * 16;
    const int batch_start = blockIdx.y * 16;
    const int tid = threadIdx.x;

    if (row_start >= total_m) return;
    if (batch_start >= N) return;

    const int my_tile_row = tid & 15;
    const int my_row = row_start + my_tile_row;
    const int safe_row = (my_row < total_m) ? my_row : (total_m - 1);

    const char* A;
    int local_row;
    if (safe_row < gate_m) {
        A = A_gate; local_row = safe_row;
    } else {
        A = A_up;   local_row = safe_row - gate_m;
    }

    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 160;
    const char* row_base = A + local_row * row_stride;

    const int safe_batch = (batch_start + my_tile_row < N)
                           ? (batch_start + my_tile_row) : 0;

    const int load_tile_row = tid >> 1;
    const int load_lo       = (tid & 1) * 8;
    const int load_my_row = row_start + load_tile_row;
    const int load_safe_row = (load_my_row < total_m) ? load_my_row : (total_m - 1);
    const char* load_A;
    int load_local_row;
    if (load_safe_row < gate_m) {
        load_A = A_gate; load_local_row = load_safe_row;
    } else {
        load_A = A_up;   load_local_row = load_safe_row - gate_m;
    }
    const char* load_row_base = load_A + (long long)load_local_row * row_stride;

    __shared__ _Float16 cb_lds[16 * 16];

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (int g = 0; g < groups_per_row; g++) {
        const __half* cb_h_load = (const __half*)(load_row_base + g * 160);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            cb_lds[load_tile_row * 16 + load_lo + i] = cb_h_load[load_lo + i];
        }
        __syncthreads();

        const char* gp = row_base + g * 160;
        const _Float16* x_base = X + (long long)safe_batch * K + g * 256;
        const unsigned char* dp = (const unsigned char*)(gp + 32);
        const int cb_base = my_tile_row * 16;

