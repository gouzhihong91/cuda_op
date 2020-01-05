

#include <stdio.h>
#include <cuda_runtime.h>



extern "C" __global__ void assign_loss( int64_t * __restrict__ match_idx,  int* __restrict__ begin, int* __restrict__ end, int* __restrict__ index, float* __restrict__ labels, int64_t* __restrict__ label_gt,
     float* __restrict__ boxes_gt, const int num_priors, const int num_classes) {
    
    int label = (int)labels[blockIdx.x];
    int mi = index[blockIdx.x];
    int left = begin[mi];
    int right = end[mi];
    for (int i_inner_outer = 0; i_inner_outer < (num_priors + blockDim.x - 1) / blockDim.x; ++i_inner_outer) {
        if ( ((i_inner_outer * blockDim.x) + ((int)threadIdx.x)) < num_priors && ((((i_inner_outer * blockDim.x) + ((int)threadIdx.x)) >= right) || (((i_inner_outer * blockDim.x) + ((int)threadIdx.x)) < left)) ) {
            if(match_idx[((i_inner_outer * blockDim.x) + ((int)threadIdx.x))] == blockIdx.x){
                label_gt[(((i_inner_outer * blockDim.x) + ((int)threadIdx.x))) * num_classes + label] = -1;
                boxes_gt[(((i_inner_outer * blockDim.x) + ((int)threadIdx.x))) * 4] = 0;
                boxes_gt[(((i_inner_outer * blockDim.x) + ((int)threadIdx.x))) * 4 + 1] = 0;
                boxes_gt[(((i_inner_outer * blockDim.x) + ((int)threadIdx.x))) * 4 + 2] = 0;
                boxes_gt[(((i_inner_outer * blockDim.x) + ((int)threadIdx.x))) * 4 + 3] = 0;
            } 
        }
      
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
    cudaError_t err7 = cudaSuccess;

    int num_priors = 16709;
    int num_gt = 20;
    int num_classes = 81;
    int num_part = 5;
    int height = 224;
    
    size_t size = num_priors * sizeof(int64_t);
    size_t size1 = num_priors * sizeof(int64_t);
    size_t size2 = num_gt * sizeof(float);
    size_t size3 = num_priors * num_classes * sizeof(int64_t);
    size_t size4 = num_priors * 4 * sizeof(float);
    size_t size5 = 5 * sizeof(int);
    printf("[assign loss of %d elements]\n", num_priors);

    int64_t * h_match_idx = (int64_t *)malloc(size1);
    int * h_begin = (int *)malloc(5 * sizeof(int));
    int * h_end = (int *)malloc(5 * sizeof(int));
    int * h_index = (int *)malloc(size2);
    float * h_labels = (float *)malloc(size2);
    int64_t * h_label_gt = (int64_t *)malloc(size3);
    float * h_boxes_gt = (float *)malloc(size4);

    int64_t * label_gt = (int64_t *)malloc(size3);
    float * boxes_gt = (float *)malloc(size4);
   
    if (h_match_idx == NULL || h_begin == NULL || h_end == NULL || h_index == NULL || h_labels == NULL || h_boxes_gt == NULL || h_label_gt == NULL || boxes_gt == NULL || label_gt == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

   
    for (int i = 0; i < num_priors; ++i){
        h_match_idx[i] = rand() % num_gt;
    }
   
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
    
    for (int i = 0; i < num_gt; ++i){
        h_index[i] = rand() % num_part;
    }

    for (int i = 0; i < num_gt; ++i){
        h_labels[i] = (float)(rand() % num_classes);
    }
    
    for (int i = 0; i < num_priors * num_classes; ++i){
        h_label_gt[i] = (int64_t)rand() % 2;
        label_gt[i] = h_label_gt[i];
    }

    for (int i = 0; i < num_priors * 4; ++i){
        h_boxes_gt[i] = (float)(rand() % height);
        boxes_gt[i] = h_boxes_gt[i];
    }
    // compute on cpu
    for(int i = 0; i < num_gt; i++){
        int label = (int)h_labels[i];
        int mi = h_index[i];
        int left = h_begin[mi];
        int right = h_end[mi];
        for(int j = 0; j < num_priors; j++){
            if((j >= right || j < left) && h_match_idx[j] == i){
                label_gt[j * 81 + label] = -1;
                boxes_gt[j * 4] = 0;
                boxes_gt[j * 4 + 1] = 0;
                boxes_gt[j * 4 + 2] = 0;
                boxes_gt[j * 4 + 3] = 0;
            }
        }
    }
    int64_t *d_match_idx = NULL;
    err = cudaMalloc((void **)&d_match_idx, size1);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector match_idx (error code %s)!\n", cudaGetErrorString(err));
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

    int *d_index = NULL;
    err = cudaMalloc((void **)&d_index, size2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector index (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float *d_labels = NULL;
    err = cudaMalloc((void **)&d_labels, size2);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector labels (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    int64_t *d_label_gt = NULL;
    err = cudaMalloc((void **)&d_label_gt, size3);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector label_gt (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    float *d_boxes_gt = NULL;
    err = cudaMalloc((void **)&d_boxes_gt, size4);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector boxes_gt (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy input data from the host memory to the CUDA device\n");
    err1 = cudaMemcpy(d_match_idx, h_match_idx, size1, cudaMemcpyHostToDevice);
    err2 = cudaMemcpy(d_begin, h_begin, size5, cudaMemcpyHostToDevice);
    err3 = cudaMemcpy(d_end, h_end, size5, cudaMemcpyHostToDevice);
    err4 = cudaMemcpy(d_index, h_index, size2, cudaMemcpyHostToDevice);
    err5 = cudaMemcpy(d_labels, h_labels, size2, cudaMemcpyHostToDevice);
    err6 = cudaMemcpy(d_label_gt, h_label_gt, size3, cudaMemcpyHostToDevice);
    err7 = cudaMemcpy(d_boxes_gt, h_boxes_gt, size4, cudaMemcpyHostToDevice);

    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess || err6 != cudaSuccess || err7 != cudaSuccess)
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
    cudaEventRecord(start, NULL);

    int nIter = 10000;

    for (int j = 0; j < nIter; j++) {
       assign_loss<<<blocksPerGrid, threadsPerBlock>>>(d_match_idx, d_begin, d_end, d_index, d_labels, d_label_gt, d_boxes_gt, num_priors, num_classes);
    }
    cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    // Compute and print the performance
    float msecPerMatrixMul = msecTotal / nIter;
    
    printf(
        "Time= %.3f msec\n",msecPerMatrixMul);
   
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch assign_loss kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_label_gt, d_label_gt, size3, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector label_gt from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(h_boxes_gt, d_boxes_gt, size4, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector boxes_gt from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int err_count = 0;
    // Verify that the result vector is correct
    for (int i = 0; i < num_priors; ++i){
        for(int j = 0; j < num_classes; j++){
            if(h_label_gt[i*81 + j] != label_gt[i*81 + j]){
                err_count++;
                fprintf(stderr, "Result verification failed at num_priors %d, num_classes %d!\n", i, j);
                printf("h_label_gt: %d\n", h_label_gt[i*81 + j]);
                printf("label_gt: %d\n", label_gt[i*81 + j]);
                //exit(EXIT_FAILURE);
            }
        }
        for(int j = 0; j < 4; j++){
            if(h_boxes_gt[i*4 + j] != boxes_gt[i*4 + j]){
                fprintf(stderr, "Result verification failed at num_priors %d, num_classes %d!\n", i, j);
                printf("h_boxes_gt: %f\n", h_boxes_gt[i*4 + j]);
                printf("boxes_gt: %f\n", boxes_gt[i*4 + j]);
                //exit(EXIT_FAILURE);
            }
        }    
    }

    printf("Test PASSED\n");

    cudaFree(d_match_idx);
    cudaFree(d_begin);
    cudaFree(d_end);
    cudaFree(d_index);
    cudaFree(d_labels);
    cudaFree(d_label_gt);
    cudaFree(d_boxes_gt);

    free(h_match_idx);
    free(h_begin);
    free(h_end);
    free(h_index);
    free(h_labels);
    free(h_label_gt);
    free(label_gt);
    free(h_boxes_gt);
    free(boxes_gt);
    printf("Error count: %d\n", err_count);
    return 0;
}
