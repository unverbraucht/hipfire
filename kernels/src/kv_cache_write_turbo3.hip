// Fused KV write: both K and V in one kernel, 32 threads per head for parallel FWHT.
// Grid: n_kv_heads blocks, 32 threads each.
extern "C" __global__ void kv_cache_write_turbo3(
    unsigned char* __restrict__ k_dst,
    unsigned char* __restrict__ v_dst,
    const float* __restrict__ k_src,
    const float* __restrict__ v_src,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_kv_heads,
    int head_dim
) {
    const int h = blockIdx.x;
    if (h >= n_kv_heads) return;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;  // 32
    const int pos = pos_buf[0];

    const int qs_bytes = head_dim / 4;
    const int sign_bytes = head_dim / 8;
    const int bytes_per_head = 4 + qs_bytes + sign_bytes;  // 52
    const int bytes_per_pos = n_kv_heads * bytes_per_head;

    extern __shared__ float smem[];
    float* x = smem;          // 128 floats for current vector
    float* scratch = smem + head_dim;  // 32 floats for reductions

    // Process K then V
    for (int kv = 0; kv < 2; kv++) {
        const float* src = (kv == 0) ? (k_src + h * head_dim) : (v_src + h * head_dim);
        unsigned char* dst = (kv == 0) ? k_dst : v_dst;
        unsigned char* out = dst + (size_t)pos * bytes_per_pos + h * bytes_per_head;

        // Load into shared memory
        for (int d = tid; d < head_dim; d += nthreads)
            x[d] = src[d];
        __syncthreads();

        // Parallel L2 norm reduction
        float local_sq = 0.0f;
        for (int d = tid; d < head_dim; d += nthreads)
            local_sq += x[d] * x[d];
        scratch[tid] = local_sq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        float orig_norm = sqrtf(scratch[0]);
        float inv_norm = (orig_norm > 1e-10f) ? 1.0f / orig_norm : 0.0f;

        // Parallel normalize
        for (int d = tid; d < head_dim; d += nthreads)
            x[d] *= inv_norm;
        __syncthreads();

        // Register-only FWHT via __shfl_xor
        const int dpt = head_dim / nthreads;
        const int d0 = tid * dpt;
        float fa = x[d0], fb = x[d0+1], fc = x[d0+2], fd = x[d0+3];
        fwht_shfl_forward(fa, fb, fc, fd, signs1, signs2, tid);

        // Quantize to 3-bit from registers
        float local_recon_sq = 0.0f;
        unsigned char local_qs[4] = {0, 0, 0, 0};
        unsigned char local_hi = 0;
        float fvals[4] = {fa, fb, fc, fd};

        for (int i = 0; i < dpt; i++) {
            float val = fvals[i];
            int idx = (val > -0.154258f) + (val > -0.091775f) + (val > -0.043589f) + (val > 0.0f)
                     + (val > 0.043589f) + (val > 0.091775f) + (val > 0.154258f);
            float c = TURBO_C3[idx];
            local_recon_sq += c * c;
            local_qs[i / 4] |= ((idx & 0x3) << ((i % 4) * 2));
            local_hi |= (((idx >> 2) & 0x1) << i);
        }

        // Reduce reconstruction norm
        scratch[tid] = local_recon_sq;
        __syncthreads();
        for (int s = 16; s > 0; s >>= 1) {
            if (tid < s) scratch[tid] += scratch[tid + s];
            __syncthreads();
        }
        float recon_norm = sqrtf(scratch[0]);
        float corrected_norm = (recon_norm > 1e-10f) ? orig_norm / recon_norm : orig_norm;

        // Write output: thread 0 writes norm, all threads write their packed bytes
        if (tid == 0) *(float*)out = corrected_norm;

        // Each thread writes 1 byte of qs (its 4 dims packed into 2-bit)
        // qs layout: bytes 4..35 (32 bytes for 128 dims, 4 per byte)
        out[4 + tid] = local_qs[0];

        // High bits: each thread has 4 bits, pack 8 threads per byte
        // Thread tid contributes bits for dims d0..d0+3
        // Byte index = tid / 2 (since 8 bits per byte, 4 bits per thread, 2 threads per byte)
        unsigned char hi_byte = local_hi;
        if (tid % 2 == 0) {
            // Even thread: low 4 bits of byte
            out[4 + qs_bytes + tid/2] = hi_byte;
        }
        __syncthreads();
        if (tid % 2 == 1) {
            // Odd thread: high 4 bits of byte
            out[4 + qs_bytes + tid/2] |= (hi_byte << 4);
        }
        __syncthreads();
    }
}
