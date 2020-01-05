
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> 
#include <ctime> 
using namespace std;

extern "C" __global__ void loss_index_1024( float* __restrict__ loss, int64_t * __restrict__ match_idx, int* __restrict__ begin, int* __restrict__ end, int * index,
const int num_priors, const int num_part) 
{
   
   __shared__ int count;
   __shared__ float red_buf0[1024];
   extern __shared__ float loss_levels_rf[];
  
   for(int i = 0; i < num_part; ++i){
     count = 0;
     loss_levels_rf[threadIdx.x + i * blockDim.x] = 0.000000e+00f;
     for (int k1_outer = 0; k1_outer < (num_priors + blockDim.x - 1) / blockDim.x; ++k1_outer) {
       if(((((begin[i] <= ((k1_outer * blockDim.x) + ((int)threadIdx.x))) && (((k1_outer * blockDim.x) + ((int)threadIdx.x)) < end[i])) && (match_idx[((k1_outer * blockDim.x) + ((int)threadIdx.x))] == ((int)blockIdx.x))))){
         loss_levels_rf[threadIdx.x + i * blockDim.x] = loss_levels_rf[threadIdx.x + i * blockDim.x] + loss[k1_outer * blockDim.x + threadIdx.x];
         atomicAdd(&count, 1);
        }
      }
      __syncthreads();
      ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = loss_levels_rf[threadIdx.x + i * blockDim.x];
      __syncthreads();
      if (((int)threadIdx.x) < 512) {
        ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.x) + 512)]);
      }
      __syncthreads();
      if (((int)threadIdx.x) < 256) {
        ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.x) + 256)]);
      }
      __syncthreads();
      if (((int)threadIdx.x) < 128) {
        ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.x) + 128)]);
      }
      __syncthreads();
      if (((int)threadIdx.x) < 64) {
        ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.x) + 64)]);
      }
      __syncthreads();
      if (((int)threadIdx.x) < 32) {
        ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.x) + 32)]);
      }
      __syncthreads();
      if (((int)threadIdx.x) < 16) {
        ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.x) + 16)]);
        ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.x) + 8)]);
        ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.x) + 4)]);
        ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.x) + 2)]);
        ((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] = (((volatile __shared__ float*)red_buf0)[((int)threadIdx.x)] + ((volatile __shared__ float*)red_buf0)[(((int)threadIdx.x) + 1)]);
      }
      __syncthreads();
      if (((int)threadIdx.x) == 0) {
        loss_levels_rf[i * blockDim.x] = ((volatile __shared__ float*)red_buf0)[0] / count; 
        //printf("blockIdx.x: %d loss: %f\n", blockIdx.x, loss_levels_rf[i]);
        //printf("blockIdx.x: %d count: %d\n", blockIdx.x, count);
      }
      __syncthreads();

    }
    if (threadIdx.x == 0) {
      //printf("blockIdx.x: %d\n", blockIdx.x);
      //printf("sum: %f %f %f %f %f\n",loss_levels_rf[0], loss_levels_rf[1], loss_levels_rf[2], loss_levels_rf[3],loss_levels_rf[4]);
      float max_loss = 0.0f;
      for(int i = 0; i < num_part; i++){
        if(loss_levels_rf[i * blockDim.x] > max_loss){
          max_loss = loss_levels_rf[i * blockDim.x];
        }
      }
     for(int i = 0; i < num_part; i++){
       if(loss_levels_rf[i * blockDim.x] < 0.000001f || isnan(loss_levels_rf[i * blockDim.x])){
         loss_levels_rf[i * blockDim.x] = max_loss + 1.0f;
        }
      }
      float min_loss = loss_levels_rf[0];
      volatile int index_min = 0;
      for(int i = 1; i < num_part; i++) {
        if(loss_levels_rf[i * blockDim.x] < min_loss){
          min_loss = loss_levels_rf[i * blockDim.x];
          index_min = i;
        }
      }
      index[blockIdx.x] = index_min;
    }
}

int
main(void)
{
  
    cudaError_t err = cudaSuccess;
    cudaError_t err1 = cudaSuccess;
    cudaError_t err2 = cudaSuccess;
    cudaError_t err3 = cudaSuccess;
    cudaError_t err4 = cudaSuccess;
    cudaError_t err5 = cudaSuccess;
    cudaError_t err6 = cudaSuccess;
    int num_priors = 16709;
    int num_gt = 20;
    int num_part = 5;
    size_t size = num_priors * num_gt * sizeof(float);
    size_t size1 = num_priors * sizeof(float);
    size_t size2 = num_gt * sizeof(int);
    size_t size3 = num_gt * 5 * sizeof(int);
    size_t size4 = num_priors * sizeof(int64_t);
    size_t size5 = 5 * sizeof(int);

    printf("[assign loss:]\n");

    float *h_loss = (float *)malloc(size1);
    int64_t *h_match_idx = (int64_t *)malloc(size4);

    float *h_loss1 = (float *)malloc(size);
    float *h_loss2 = (float *)malloc(size);
    float *h_loss3 = (float *)malloc(size);
    float *h_loss4 = (float *)malloc(size);
    float *h_loss5 = (float *)malloc(size);

    int *h_index = (int *)malloc(size2);
    int *h_count = (int *)malloc(size3);
    
    if (h_loss == NULL || h_match_idx == NULL || h_loss1 == NULL || h_loss2 == NULL || h_loss3 == NULL || h_loss4 == NULL || h_loss5 == NULL || h_index == NULL || h_count == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    int * h_begin;
    int * h_end;
    h_begin = new int[5];
    h_end = new int[5];
    h_begin[0] = 0;
    h_end[0] =  12544;

    h_begin[1] = 12544;
    h_end[1] = 12544 + 3136;

    h_begin[2] = 12544 + 3136;
    h_end[2] = 12544 + 3136 + 784;

    h_begin[3] = 12544 + 3136 + 784;
    h_end[3] = 12544 + 3136 + 784 + 196;

    h_begin[4] = 12544 + 3136 + 784 + 196;
    h_end[4] = num_priors;

    for (int i = 0; i < num_priors; ++i){
        h_match_idx[i] = rand() % num_gt;
    }

    for (int i = 0; i < num_gt * 5; ++i){
        h_count[i] = 0;
    }

    unsigned seed;  
    
    seed = time(0);
    srand(seed);

    for (int i = 0; i < num_priors; ++i){
        h_loss[i] = (float)rand() / (RAND_MAX + 1.0);
    }

    for(int j = 0; j < num_gt; j++){
        for (int i = 0; i < num_priors; ++i){
            h_loss1[j * 16709 + i] = 0.0;
            h_loss2[j * 16709 + i] = 0.0;
            h_loss3[j * 16709 + i] = 0.0;
            h_loss4[j * 16709 + i] = 0.0;
            h_loss5[j * 16709 + i] = 0.0;
        }
    }
    for(int j = 0; j < num_gt; j++){
        for (int i = 0; i < num_priors; ++i){
            if(h_match_idx[i] == j){
                if(i >= h_begin[0] && i < h_end[0]){
                    h_loss1[j * 16709 + i] = h_loss[i];
                    h_count[j*5] += 1;
                }
                if(i >= h_begin[1] && i < h_end[1]){
                    h_loss2[j * 16709 + i] = h_loss[i];
                    h_count[j*5 + 1] += 1;
                }
                if(i >= h_begin[2] && i < h_end[2]){
                    h_loss3[j * 16709 + i] = h_loss[i];
                    h_count[j*5 + 2] += 1;
                }
                if(i >= h_begin[3] && i < h_end[3]){
                    h_loss4[j * 16709 + i] = h_loss[i];
                    h_count[j*5 + 3] += 1;
                }
                if(i >= h_begin[4] && i < h_end[4]){
                    h_loss5[j * 16709 + i] = h_loss[i];
                    h_count[j*5 + 4] += 1;
                }
            }
        }
    }

    // Allocate the device output vector C
    float *d_loss = NULL;
    err = cudaMalloc((void **)&d_loss, size1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector loss (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    int64_t *d_match_idx = NULL;
    err = cudaMalloc((void **)&d_match_idx, size4);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector match_idx (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    

    int *d_index = NULL;
    err = cudaMalloc((void **)&d_index, size2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector index (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    int *d_begin = NULL;
    err = cudaMalloc((void **)&d_begin, size5);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector begin (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    int *d_end = NULL;
    err = cudaMalloc((void **)&d_end, size5);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector end (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Copy input data from the host memory to the CUDA device\n");
    err1 = cudaMemcpy(d_loss, h_loss, size1, cudaMemcpyHostToDevice);
    err2 = cudaMemcpy(d_match_idx, h_match_idx, size4, cudaMemcpyHostToDevice);
    err3 = cudaMemcpy(d_begin, h_begin, size5, cudaMemcpyHostToDevice);
    err4 = cudaMemcpy(d_end, h_end, size5, cudaMemcpyHostToDevice);
    err5 = cudaMemcpy(d_index, h_index, size2, cudaMemcpyHostToDevice);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    int threadsPerBlock = 1024;
    int blocksPerGrid = num_gt;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    
    
    int nIter = 10000;
    float msecTotal = 0.0f;

    for (int j = 0; j < nIter; j++) {
     
      cudaEventRecord(start, NULL);
      loss_index_1024<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * num_part * sizeof(float)>>>(d_loss, d_match_idx, d_begin, d_end, d_index, num_priors, num_part);
      cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&msecTotal, start, stop);

      // Compute and print the performance
     //float msecPerMatrixMul = msecTotal / nIter;
     float msecPerMatrixMul = msecTotal;
      printf(
          "Time= %.3f msec\n",msecPerMatrixMul);
    }
  
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch loss_index kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_index, d_index, size2, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector h_index from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Verify that the result vector is correct
    for (int i = 0; i < num_gt; ++i){
        float sum[5] = {0.0};
        float max_loss = 0.0;
        for(int j = 0; j < num_priors; j++){
            sum[0] += h_loss1[i * num_priors + j];
            sum[1] += h_loss2[i * num_priors + j];
            sum[2] += h_loss3[i * num_priors + j];
            sum[3] += h_loss4[i * num_priors + j];
            sum[4] += h_loss5[i * num_priors + j];
        }
        for(int j = 0; j < 5; j++){
            sum[j] /= h_count[i * 5 + j];
        }
        for(int j = 0; j < 5; j++){
            if(sum[j] > max_loss){
                max_loss = sum[j];
            }
         }
        for(int j = 0; j < 5; j++){
             if(sum[j] < 0.000001f){
                 sum[j] = max_loss + 1.0f;
             }
        }
        float min_loss = sum[0];
        int index = 0;
        for(int j = 1; j < 5; j++) {
             if(sum[j] < min_loss){
                 min_loss = sum[j];
                 index = j;
             }
        }    
        if (index != h_index[i])
        {
            fprintf(stderr, "Result verification failed at num_gt %d!\n", i);
            printf("index: %d\n", index);
            printf("count: %d %d %d %d %d\n", h_count[i * 5], h_count[i * 5 + 1], h_count[i * 5 + 2], h_count[i * 5 + 3],h_count[i * 5 + 4]);
            printf("sum: %f %f %f %f %f\n", sum[0], sum[1], sum[2], sum[3],sum[4]);
            printf("gpu_index: %d\n",h_index[i]);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    cudaFree(d_match_idx);
    cudaFree(d_loss);
    cudaFree(d_begin);
    cudaFree(d_end);
    cudaFree(d_index);
    
    free(h_loss);
    free(h_loss1);
    free(h_loss2);
    free(h_loss3);
    free(h_loss4);
    free(h_loss5);
    free(h_begin);
    free(h_end);
    free(h_index);
    free(h_match_idx);
    free(h_count);
    printf("Done\n");
    return 0;
}
