extern "C" __global__ void loss_index_1024( float* __restrict__ A, float * __restrict__ max, unsigned int * __restrict__ max_idx, 
    const int num_gt, const int num_anchor) 
    {
       __shared__ float red_buf0[1024];
       __shared__ float index_buf0[1024];

       float compute_red_rf[1];
       unsigned int  index[1];
       compute_red_rf[0] = -3.402823e+38f;
       index[0] = 0;
       int start = blockIdx.x * num_anchor;

       for (int k1_outer = 0; k1_outer < (num_anchor + 1023) / 1024; ++k1_outer) {
         if ( k1_outer * 1024 + threadIdx.x < num_anchor) {
             if(compute_red_rf[0] <  A[start + k1_outer * 1024 + threadIdx.x]){
                 compute_red_rf[0] = A[start + k1_outer * 1024 + threadIdx.x];
                 index[0] = k1_outer * 1024 + threadIdx.x;
                }
            }
        }
        __syncthreads();
        ((volatile __shared__ float*)red_buf0)[threadIdx.x] = compute_red_rf[0];
        ((volatile __shared__ float*)index_buf0)[threadIdx.x] = index[0];
        __syncthreads();

        if (((int)threadIdx.x) < 512) {
            if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 512]){
                red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 512];
                index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 512];
            }
        }
        __syncthreads();

        if (((int)threadIdx.x) < 256) {
            if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 256]){
                red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 256];
                index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 256];
            }
        }
        __syncthreads();
        if (((int)threadIdx.x) < 128) {
            if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 128]){
                red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 128];
                index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 128];
            }
            
        }
        __syncthreads();
        if (((int)threadIdx.x) < 64) {
            if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 64]){
                red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 64];
                index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 64];
            }
        }
        __syncthreads();
        if (((int)threadIdx.x) < 32) {
            if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 32]){
                red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 32];
                index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 32];
            }
        }
        __syncthreads();
        
        if (((int)threadIdx.x) < 16) {
            if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 16]){
                red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 16];
                index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 16];
            }
            if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 8]){
                red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 8];
                index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 8];
            }
            if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 4]){
                red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 4];
                index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 4];
            }
            if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 2]){
                red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 2];
                index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 2];
            }
            if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 1]){
                red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 1];
                index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 1];
            }
        }
        __syncthreads();
        if (((int)threadIdx.x) == 0) {
            max[blockIdx.x] = ((volatile __shared__ float*)red_buf0)[0]; 
            max_idx[blockIdx.x] = ((volatile __shared__ float*)index_buf0)[0]; 
        }    
    }