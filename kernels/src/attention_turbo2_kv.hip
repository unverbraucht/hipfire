extern "C" __global__ void attention_turbo2_kv(
    const float* __restrict__ q,
    const unsigned char* __restrict__ k_cache,
    const unsigned char* __restrict__ v_cache,
    float* __restrict__ out,
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,
    const float* __restrict__ signs2,
    int n_heads, int n_kv_heads, int head_dim, int max_seq,
    float scale_attn
) {
    int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int qs_bytes = head_dim / 4;
    const int bytes_per_head = 4 + qs_bytes;
    const int bytes_per_pos = n_kv_heads * bytes_per_head;
    const int kv_head_off = kv_h * bytes_per_head;
    const int dpt = head_dim / nthreads;
    const int d0 = tid * dpt;

    float* scores = sdata;  // only scores in shared memory now

    // Register-only FWHT on Q (zero shared memory, zero barriers)
    float mq0 = q[h * head_dim + d0];
    float mq1 = q[h * head_dim + d0 + 1];
    float mq2 = q[h * head_dim + d0 + 2];
    float mq3 = q[h * head_dim + d0 + 3];
    fwht_shfl_forward(mq0, mq1, mq2, mq3, signs1, signs2, tid);
    float mq[4] = {mq0, mq1, mq2, mq3};

    // K dot products — 2-bit: 4 values per byte, trivial extraction
    for (int t = 0; t < seq_len; t++) {
        const unsigned char* kb = k_cache + (size_t)t * bytes_per_pos + kv_head_off;
        float cnorm = *(const float*)kb;
        float partial = 0.0f;
        unsigned char packed = kb[4 + tid];  // one byte = 4 dims for this thread
        partial += TURBO_C2[(packed)      & 0x3] * mq[0];
        partial += TURBO_C2[(packed >> 2) & 0x3] * mq[1];
        partial += TURBO_C2[(packed >> 4) & 0x3] * mq[2];
        partial += TURBO_C2[(packed >> 6) & 0x3] * mq[3];
        partial *= cnorm;
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor(partial, off);
        if (tid == 0) scores[t] = partial * scale_attn;
    }
    __syncthreads();

    // Parallel softmax via warp shuffle
    float lmx = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) lmx = fmaxf(lmx, scores[t]);
    for (int off = 16; off > 0; off >>= 1) lmx = fmaxf(lmx, __shfl_xor(lmx, off));
    float mx = lmx;
    float lsm = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - mx); scores[t] = e; lsm += e;
    }
    for (int off = 16; off > 0; off >>= 1) lsm += __shfl_xor(lsm, off);
    __syncthreads();
    float inv_sum = 1.0f / lsm;
    for (int t = tid; t < seq_len; t += nthreads) scores[t] *= inv_sum;
    __syncthreads();

    // V weighted sum — each thread owns 4 dims
    float mv[4] = {0, 0, 0, 0};
    for (int t = 0; t < seq_len; t++) {
        float w = scores[t];
        if (w < 1e-8f) continue;
        const unsigned char* vb = v_cache + (size_t)t * bytes_per_pos + kv_head_off;
        float wn = w * *(const float*)vb;
        unsigned char packed = vb[4 + tid];
        mv[0] += wn * TURBO_C2[(packed)      & 0x3];
        mv[1] += wn * TURBO_C2[(packed >> 2) & 0x3];
        mv[2] += wn * TURBO_C2[(packed >> 4) & 0x3];
        mv[3] += wn * TURBO_C2[(packed >> 6) & 0x3];
    }

    // Register-only inverse FWHT (zero shared memory, zero barriers)
    fwht_shfl_inverse(mv[0], mv[1], mv[2], mv[3], signs1, signs2, tid);

    float* oh = out + h * head_dim;
    oh[d0] = mv[0]; oh[d0+1] = mv[1]; oh[d0+2] = mv[2]; oh[d0+3] = mv[3];
}
