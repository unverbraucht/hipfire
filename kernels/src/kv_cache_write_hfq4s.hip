extern "C" __global__ void kv_cache_write_hfq4s(
    unsigned char* __restrict__ dst,
    const float* __restrict__ src,
    const int* __restrict__ pos_buf,
    int n_kv_heads, int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int pos = pos_buf[0];
    const float* head_src = src + h * head_dim;
    const int d0 = tid * 4;

    // Load with sign flip decorrelation
    float v0 = head_src[d0]     * TURBO_SIGNS1[d0];
    float v1 = head_src[d0 + 1] * TURBO_SIGNS1[d0 + 1];
    float v2 = head_src[d0 + 2] * TURBO_SIGNS1[d0 + 2];
    float v3 = head_src[d0 + 3] * TURBO_SIGNS1[d0 + 3];

    // Warp min/max
    float local_min = fminf(fminf(v0, v1), fminf(v2, v3));
    float local_max = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_min = fminf(local_min, __shfl_xor(local_min, offset));
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset));
    }

    float scale = (local_max - local_min) / 15.0f;
    float zero = local_min;
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    int bytes_per_block = 8 + head_dim / 2;
    int bytes_per_pos = n_kv_heads * bytes_per_block;
    unsigned char* out = dst + (size_t)pos * bytes_per_pos + h * bytes_per_block;

    if (tid == 0) {
        *(float*)(out) = scale;
        *(float*)(out + 4) = zero;
    }

    int q0 = __float2int_rn((v0 - zero) * inv_scale); q0 = max(0, min(15, q0));
    int q1 = __float2int_rn((v1 - zero) * inv_scale); q1 = max(0, min(15, q1));
    int q2 = __float2int_rn((v2 - zero) * inv_scale); q2 = max(0, min(15, q2));
    int q3 = __float2int_rn((v3 - zero) * inv_scale); q3 = max(0, min(15, q3));

    out[8 + tid * 2]     = (unsigned char)((q1 << 4) | q0);
    out[8 + tid * 2 + 1] = (unsigned char)((q3 << 4) | q2);
}
