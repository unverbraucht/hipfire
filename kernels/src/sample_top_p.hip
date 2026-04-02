#include <hip/hip_runtime.h>

extern "C" __global__ void sample_top_p(
    float* __restrict__ logits,          // [vocab_size] (modified in-place for repeat penalty)
    unsigned int* __restrict__ result,   // [2]: token_id, rng_state
    const unsigned int* __restrict__ repeat_tokens, // [repeat_window] recent token IDs
    int vocab_size,
    float temperature,
    float top_p,
    unsigned int rand_seed,
    int repeat_window,
    float repeat_penalty
) {
    extern __shared__ char smem[];
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Shared memory layout:
    // [0..nthreads-1]: reduction workspace (floats)
    // [nthreads..nthreads+MAX_CAND-1]: candidate scores (floats)
    // [nthreads+MAX_CAND..nthreads+2*MAX_CAND-1]: candidate indices (ints)
    // [nthreads+2*MAX_CAND]: candidate count (int)
    const int MAX_CAND = 512;
    float* reduce = (float*)smem;
    float* cand_scores = reduce + nthreads;
    int* cand_indices = (int*)(cand_scores + MAX_CAND);
    int* cand_count = cand_indices + MAX_CAND;

    // Phase 0: Apply repetition penalty — only touches repeat_window positions
    for (int j = tid; j < repeat_window; j += nthreads) {
        int tok_id = repeat_tokens[j];
        if (tok_id < vocab_size) {
            float v = logits[tok_id];
            logits[tok_id] = (v > 0.0f) ? (v / repeat_penalty) : (v * repeat_penalty);
        }
    }
    __syncthreads();

    // Phase 1: Find max logit
    float local_max = -1e30f;
    for (int i = tid; i < vocab_size; i += nthreads) {
        local_max = fmaxf(local_max, logits[i]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
        __syncthreads();
    }
    float max_val = reduce[0];

    // Phase 2: Collect candidates above threshold
    float inv_temp = 1.0f / temperature;
    float thresh = max_val - 30.0f * temperature;

    if (tid == 0) *cand_count = 0;
    __syncthreads();

    for (int i = tid; i < vocab_size; i += nthreads) {
        if (logits[i] >= thresh) {
            int pos = atomicAdd(cand_count, 1);
            if (pos < MAX_CAND) {
                cand_scores[pos] = (logits[i] - max_val) * inv_temp;
                cand_indices[pos] = i;
            }
        }
    }
    __syncthreads();

    // Phase 3: Thread 0 — softmax, sort top-20, top-p filter, sample
    if (tid == 0) {
        int n = *cand_count;
        if (n > MAX_CAND) n = MAX_CAND;
        if (n == 0) { result[0] = 0; result[1] = rand_seed; return; }

        // Exp + sum
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            float p = expf(cand_scores[i]);
            cand_scores[i] = p;
            sum += p;
        }

        // Partial selection sort — find top-20 only
        const int TOP_K = 20 < n ? 20 : n;
        for (int i = 0; i < TOP_K; i++) {
            int best = i;
            for (int j = i + 1; j < n; j++) {
                if (cand_scores[j] > cand_scores[best]) best = j;
            }
            if (best != i) {
                float ts = cand_scores[i]; cand_scores[i] = cand_scores[best]; cand_scores[best] = ts;
                int ti = cand_indices[i]; cand_indices[i] = cand_indices[best]; cand_indices[best] = ti;
            }
        }

        // RNG: xorshift32
        unsigned int rng = rand_seed;
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        float r = (float)rng / 4294967295.0f * sum;

        // Top-p + sample in one pass
        float cum = 0.0f;
        float p_thresh = top_p * sum;
        for (int i = 0; i < TOP_K; i++) {
            cum += cand_scores[i];
            if (cum >= r) {
                result[0] = (unsigned int)cand_indices[i];
                result[1] = rng;
                return;
            }
            if (cum >= p_thresh) break;
        }
        // Fallback: top-1
        result[0] = (unsigned int)cand_indices[0];
        result[1] = rng;
    }
}
