        __syncthreads();
    }

    const int out_col = batch_start + my_tile_row;
    if (out_col < N) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < total_m) {
                float* Y;
                int proj_row;
                int out_stride;
                if (out_row < gate_m) {
                    Y = Y_gate; proj_row = out_row;            out_stride = gate_m;
                } else {
                    Y = Y_up;   proj_row = out_row - gate_m;   out_stride = up_m;
                }
                Y[(long long)out_col * out_stride + proj_row] = acc[j];
            }
        }
    }
}
