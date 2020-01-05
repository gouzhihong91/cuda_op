#include <cstdio>
#include <stdio.h>

#include <curand.h>
#include <cmath>
#include <iostream>
#include <cublas_v2.h>
#include <curand_kernel.h>

#define ErrChk(code) { Assert((code), __FILE__, __LINE__); }
inline void Assert(cublasStatus_t  code, const char *file, int line){
    if (code != CUBLAS_STATUS_SUCCESS )
    {
        std::cout << "cublas API Error: " << code << " " <<  file << "@" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void Assert(cudaError_t  code, const char *file, int line){
    if (code != cudaSuccess)
    {
        std::cout << "CUDA API Error: " << code << " " <<  file << "@" << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

// cuda_runtime.h is contained in curand.h

#define CUDA_KERNEL_LOOP(i, n)                                                 \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
     i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 256;

inline int GET_BLOCKS(const int N) {
return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}
__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}
__global__ void rand_offset(curandState * state, float* offset, int n){
    //printf("IN");
    CUDA_KERNEL_LOOP(index, n){    
        offset[index] = (curand_normal(state + threadIdx.x));
    }
}
__global__ void rand_input(curandState * state, float* offset, int n){
    //printf("IN");
    CUDA_KERNEL_LOOP(index, n){
        offset[index] = (curand_normal(state + threadIdx.x)) * 4.0;
    }
}

__global__ void rand_weight(curandState * state, float* offset, int n){
  //printf("IN");
  CUDA_KERNEL_LOOP(index, n){
      offset[index] = (curand_normal(state + threadIdx.x)) * 0.3;
  }
}

template <typename DType>
__device__ DType deformable_im2col_bilinear(const DType* bottom_data, const int data_width,
  const int height, const int width, DType h, DType w) {

  int h_low = floorf(h);
  int w_low = floorf(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  DType lh = h - h_low;
  DType lw = w - w_low;
  DType hh = 1 - lh, hw = 1 - lw;

  DType val =
    +((h_low >= 0 && w_low >= 0)? bottom_data[h_low * data_width + w_low]*hh*hw:0)
  
    +((h_low >=0 && w_high <= width - 1)? bottom_data[h_low * data_width + w_high]*hh*lw:0)
  
    +((h_high <= height - 1 && w_low >= 0)?bottom_data[h_high * data_width + w_low]*lh*hw:0)
    +((h_high <= height - 1 && w_high <= width - 1)?bottom_data[h_high * data_width + w_high]*lh*lw:0);
  
  // DType w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  // DType val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename DType>
__global__ void deformable_im2col_gpu_kernel(const int n, const DType* data_im, const DType* data_offset,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int channel_per_deformable_group,
  const int batch_size, const int num_channels, const int deformable_group,
  const int height_col, const int width_col,
  DType* data_col) {
  CUDA_KERNEL_LOOP(index, n) { 
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    // compute deformable group index
    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;
    DType* data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    //const DType* data_im_ptr = data_im + ((b_col * num_channels + c_im) * height + h_in) * width + w_in;
    const DType* data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const DType* data_offset_ptr = data_offset + (b_col) * 2 * kernel_h * kernel_w * height_col * width_col;


    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const DType offset_h = data_offset_ptr[data_offset_h_ptr];
        const DType offset_w = data_offset_ptr[data_offset_w_ptr];
        DType val = static_cast<DType>(0);
        const DType h_im = h_in + i * dilation_h + offset_h;
        const DType w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          //const DType map_h = i * dilation_h + offset_h;
          //const DType map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = deformable_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = deformable_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename DType>
__global__ void col2im_128(float *im, float *col, int N, int K, int P, int Q){
    // N : batch  K: filter P: output_h  Q: output_w

    //Each block is responsible for P*Q element, so we need K*N blocks
  
    int k = blockIdx.x % K;
    int n = blockIdx.x / K;
  
    int in_offset = P*Q*n + k*P*Q*N; 
    int out_offset = n*K*P*Q + k*P*Q;
  
    for(int j=0; j<(P*Q)/128; ++j)
      im[out_offset+threadIdx.x+j*128] = col[in_offset+threadIdx.x+j*128];
    
    if (((P*Q)%128)!=0 && threadIdx.x<((P*Q)%128))
      im[out_offset+threadIdx.x+((P*Q)/128)*128] = col[in_offset+threadIdx.x+((P*Q)/128)*128];
}
  

int main(){
    
    int batch = 64;
    int channels = 32;
    int height = 74, width = 81;
    int pad_h = 2;
    int pad_w = 2;
    int stride_h = 2;
    int stride_w = 2;
    int dilation_h = 2;
    int dilation_w = 2;
    int kh = 7, kw = 7;

    int deformable_group = 1;
    // c % deformable_group == 0, group deformable-conv is not supported now
    int channel_per_deformable_group = channels / deformable_group;
    int out_h = (height + 2 * pad_h - (dilation_h * (kh - 1) + 1)) / stride_h + 1;
    int out_w = (width + 2 * pad_w - (dilation_w * (kw - 1) + 1)) / stride_w + 1;

    int num_filter = 26;

    int seed = (unsigned)time(NULL);
   
    float* input, *offset, *data_col, *weight, *out_col, *out;
    float *h_out_col;
    float *h_out;
    h_out_col = (float*)malloc(num_filter *  batch * out_h * out_w * sizeof(float));
    h_out = (float*)malloc(num_filter *  batch * out_h * out_w * sizeof(float));

    for(int i = 0; i < num_filter *  batch * out_h * out_w; i++){
      h_out_col[i] = 0.0;
    }
    curandState* devStates;
    cudaMalloc (&devStates, CUDA_NUM_THREADS * sizeof(curandState));
    
    cudaMalloc(&input, batch * channels * height * width * sizeof(float));
    cudaMalloc(&offset, batch * kh*kw*2 * out_h * out_w * sizeof(float));
    //  data_offset pointer of offsets (N, deformable_group*kernel_h*kernel_w*2, OH, OW, ...) in the offset batch
    ErrChk(cudaMalloc(&data_col, channels*kh*kw *batch*out_h*out_w *sizeof(float)));
    // data_col column buffer pointer C*kh*kw*  N*OH*OW
    ErrChk(cudaMalloc(&weight, num_filter * channels * kh * kw * sizeof(float)));
    ErrChk(cudaMalloc(&out_col, num_filter * batch * out_h * out_w * sizeof(float)));
    ErrChk(cudaMalloc(&out, num_filter * batch * out_h * out_w * sizeof(float)));

    ErrChk(cudaMemcpy(out_col, h_out_col, num_filter * batch * out_h * out_w *sizeof(float), cudaMemcpyHostToDevice));

    int num_offset = batch* kh*kw*2 * out_h * out_w;
    int num_input = batch * channels * height * width;
    int num_weight = num_filter * channels * kh * kw;
    printf("num_offset: %d\n", num_offset);
    // set the seed for each thread
    // initial by cu_rand_normal
    setup_kernel<<<GET_BLOCKS(num_input), CUDA_NUM_THREADS>>>(devStates, seed);
    rand_input<<<GET_BLOCKS(num_input), CUDA_NUM_THREADS>>>(devStates, input, num_input);
    rand_offset<<<GET_BLOCKS(num_offset), CUDA_NUM_THREADS>>>(devStates, offset, num_offset);
    rand_weight<<<GET_BLOCKS(num_weight), CUDA_NUM_THREADS>>>(devStates, weight, num_weight);

    int num_thread = batch * channels * out_h * out_w; // each thread is responsible for kh*kw elements
    int thread_per_block = 256;
    int num_block = (num_thread + thread_per_block - 1) / thread_per_block;

    
    float msecTotal = 0.0f;
    float msecPerkernel;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
   
    cudaEventRecord(start, 0);
    deformable_im2col_gpu_kernel<float><<<num_block, thread_per_block>>>(
        num_thread, input, offset, height, width, kh, kw, pad_h,
        pad_w, stride_h, stride_w, dilation_h, dilation_w,
        channel_per_deformable_group, batch, channels, deformable_group, out_h, out_w, data_col);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
   
    msecPerkernel = msecTotal / 1.0;
    printf("Time= %.3f msec\n", msecPerkernel);

    cudaFree(input);
    cudaFree(offset);

    int M = num_filter;
    int K = kh * kw * channels;
    int N = batch * out_h * out_w;
    printf("M: %d\n", M);
    printf("K: %d\n", K);
    printf("N: %d\n", N);
    float alpha     = 1.0;
    float beta      = 0.0;

    // CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEventRecord(start, 0);

    int ITERATION = 1;
    for(int i=0 ;i < ITERATION; ++i)
    ErrChk(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, (void*) &alpha, (void*) data_col, CUDA_R_32F, N, (void*) weight, CUDA_R_32F, K, (void*) &beta, (void*) out_col, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  
    
    ErrChk(cudaEventRecord(stop, 0));
    ErrChk(cudaEventSynchronize(stop));
    ErrChk(cudaEventElapsedTime(&msecTotal, start, stop));

    float avg_ms = msecTotal / ITERATION;
    float num_flops      = 2 * int64_t(M) * int64_t(N) * int64_t(K) /1024.0/1024.0/1024.0;
    float gflops_per_sec   = float(num_flops) / avg_ms * 1.0e3;
    printf("Avg runtime: %.3f ms, total gflops: %0.6f, GFLOP/s: %.6f\n", avg_ms, num_flops, gflops_per_sec);

    
    cudaFree(data_col);
    cudaFree(weight);
    cudaEventRecord(start, 0);

    col2im_128<float><<<num_filter * batch, 128>>>(out, out_col, batch, num_filter, out_h, out_w);

    ErrChk(cudaEventRecord(stop, 0));
    ErrChk(cudaEventSynchronize(stop));
    ErrChk(cudaEventElapsedTime(&msecTotal, start, stop));
    
    msecPerkernel = msecTotal;
    printf("Time= %.3f msec\n", msecPerkernel);

    cudaMemcpy(h_out, out, num_filter * batch * out_h * out_w *sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(out_col);
    cudaFree(out);
    cublasDestroy(handle);
    free(h_out_col);
    free(h_out);
    cudaFree(devStates);
}