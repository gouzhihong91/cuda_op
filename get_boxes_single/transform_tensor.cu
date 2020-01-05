#include <iostream>
#include <cstdlib>
#include "cudnn.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <malloc.h>
#include <stdio.h>
#include <vector>
#include <iomanip>
#include <fstream>

#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
using namespace std;

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define ErrChk(code) { Assert((code), __FILE__, __LINE__); }
inline void Assert(cudnnStatus_t  code, const char *file, int line){
  if (code != CUDNN_STATUS_SUCCESS )
  { 
      std::cout << "cudnn API Error: " << code << " " <<  file << "@" << line << std::endl;
      exit(EXIT_FAILURE);
  }
}

inline void Assert(cudaError_t  code, const char *file, int line){
  if (code != cudaSuccess)\
  {
      std::cout << "CUDA API Error: " << code << " " <<  file << "@" << line << std::endl;
      exit(EXIT_FAILURE);\
  }
}

#define KernelErrChk                                                    \
{                                                                     \
    cudaError_t errSync = cudaGetLastError();                           \
    cudaError_t errAsync = cudaDeviceSynchronize();                     \
    if (errSync != cudaSuccess) {                                       \
      printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));   \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
    if (errAsync != cudaSuccess) {                                      \
      printf("Async kernel error: %s\n", cudaGetErrorString(errAsync)); \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
}

extern "C" __global__ void slice_2d_1024( float* __restrict__ array_slice,  float* __restrict__ array,  int* __restrict__ index, int small_dim, int dim2) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < small_dim) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < small_dim * dim2) {
      array_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = array[( (index[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2))] * 4) + (((int)threadIdx.x) & 3) )];
    }
  }
}

extern "C" __global__ void slice_2d_1024_2( float* __restrict__ array_slice,  float* __restrict__ array,  int* __restrict__ index, int small_dim, int dim2) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < small_dim * dim2) {
    array_slice[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = array[((index[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 5)] * 5) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 5))];
  }
}

extern "C" __global__ void delta2box_1( float* __restrict__ rois,  float* __restrict__ deltas,  float* __restrict__ gbox, int small_dim, float max_ratio) {
  float pbox[3];
  float denorm_deltas[1];
 for (int j = 0; j < 3; ++j) {
   if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < small_dim) {
     if ((j + (((int)threadIdx.x) & 3)) < 4) {
       pbox[j] = ((1 < (j + (((int)threadIdx.x) & 3))) ? ((rois[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) + j)] - rois[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) + j) - 2)]) + 1.000000e+00f) : ((rois[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) + j)] + rois[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) + j) + 2)]) * 5.000000e-01f));
     }
   }
 }
 if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < small_dim) {
   //denorm_deltas[0] = (( deltas[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * stds[threadIdx.x & 3] ) + mean[threadIdx.x & 3]);
   denorm_deltas[0] = (( deltas[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * 1.0 ) );
 }
 if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < small_dim) {
   denorm_deltas[0] = (((1 < (((int)threadIdx.x) & 3)) && (denorm_deltas[0] > max_ratio)) ? max_ratio : denorm_deltas[0]);
   denorm_deltas[0] = (((1 < (((int)threadIdx.x) & 3)) && (denorm_deltas[0] < -1.0*max_ratio)) ? -1.0*max_ratio : denorm_deltas[0]);
 }
 if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < small_dim) {
   if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < small_dim *4) {
     gbox[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((1 < (((int)threadIdx.x) & 3)) ? (pbox[0] * __expf(denorm_deltas[0])) : (pbox[0] + (pbox[2] * denorm_deltas[0])));
   }
 }
}

extern "C" __global__ void delta2box_2( float* __restrict__ gbox,  float* __restrict__ box, int small_dim, int width, int height) {
  float gbox1[1];
  float temp[1];
 if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < 1000) {
  gbox1[0] = (((((int)threadIdx.x) & 3) < 2) ? ((gbox[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] - (gbox[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) + 2)] * 5.000000e-01f)) + 5.000000e-01f) : ((gbox[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) - 2)] + (gbox[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] * 5.000000e-01f)) - 5.000000e-01f));
 }
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < small_dim) {
   if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < small_dim *4) {
     if(threadIdx.x % 2 == 0)
     {
      temp[0] =  (height-1) < gbox1[0]  ? height -1 : gbox1[0];
      box[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =  temp[0] < 0  ? 0 : temp[0];
     }
     else{
      temp[0] =  (width-1) < gbox1[0]  ? width -1 : gbox1[0];
      box[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] =  temp[0] < 0  ? 0 : temp[0]; 
     }
   }
 }
}

const int threadsPerBlock = sizeof(unsigned long long) * 8; 
 
__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return interS / (Sa + Sb - interS);
}
 
__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x; //block列号
 
  const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);
  //min()的目的是防止从dev_boxes中读取数据越界（原因是n_boxes不一定被threadsPerBlock整除）
  //实际上只有最后一个block中所需要的线程数目可能小于threadsPerBlock，其余均等于threadsPerBlock

  __shared__ float block_boxes[threadsPerBlock * 5]; //共享内存 每个block64个候选框
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads(); //同步线程
  
  //以下代码实现某一边界框与其余所有边界框（删去了部分重复）进行交并比的阈值判断
  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    //如果当前边界框所处的block与要比较的边界框所处的block相同，则start不从0开始，减少重复计算

    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5) > nms_overlap_thresh) {
        t |= 1ULL << i; //1ULL = unsigned long long型的数字1（最高位为第64位）
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
 
void nms(int* keep_out, int *num_out, float* boxes_host, int boxes_num, int boxes_dim, float nms_overlap_thresh) {
    
    float* boxes_dev = NULL;
    unsigned long long* mask_dev = NULL;
    const int col_blocks = DIVUP(boxes_num, threadsPerBlock); 
    
    cudaMalloc(&boxes_dev, boxes_num * boxes_dim * sizeof(float)); 
    cudaMemcpy(boxes_dev, boxes_host, boxes_num * boxes_dim * sizeof(float), cudaMemcpyHostToDevice); 
    
    cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(unsigned long long)); 

    dim3 blocks(DIVUP(boxes_num, threadsPerBlock),DIVUP(boxes_num, threadsPerBlock)); 
    dim3 threads(threadsPerBlock); 
    
    nms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, boxes_dev, mask_dev); 
    
    std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
    cudaMemcpy(&mask_host[0], mask_dev, sizeof(unsigned long long) * boxes_num * col_blocks, cudaMemcpyDeviceToHost); 
    std::vector<unsigned long long> remv(col_blocks); 
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks); 


   int num_to_keep = 0;
   for (int i = 0; i < boxes_num; i++) {
       int nblock = i / threadsPerBlock; 
       int inblock = i % threadsPerBlock;
       if (!(remv[nblock] & (1ULL << inblock))) { 
        keep_out[num_to_keep++] = i; 
        unsigned long long *p = &mask_host[0] + i * col_blocks;
        for (int j = nblock; j < col_blocks; j++) {
          remv[j] |= p[j];
        }
      }
    }
    *num_out = num_to_keep;
    cudaFree(boxes_dev);
    cudaFree(mask_dev);
}

extern "C" __global__ void box_and_score_1024( float* __restrict__ proposal,  float* __restrict__ box,  float* __restrict__ score, int keep) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < keep * 5) {
    proposal[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 5) < 4) ? box[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 5) * 4) + (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % 5))] : score[(((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / 5)]);
  }
}

char cls_name[5][100] = { 
  "/home/gouzhihong/project/data/get_box_single_cls0.txt",
  "/home/gouzhihong/project/data/get_box_single_cls1.txt",
  "/home/gouzhihong/project/data/get_box_single_cls2.txt",
  "/home/gouzhihong/project/data/get_box_single_cls3.txt",
  "/home/gouzhihong/project/data/get_box_single_cls4.txt",
};

char anchor_name[5][100] ={
  "/home/gouzhihong/project/data/get_box_single_anchor0.txt",
  "/home/gouzhihong/project/data/get_box_single_anchor1.txt",
  "/home/gouzhihong/project/data/get_box_single_anchor2.txt",
  "/home/gouzhihong/project/data/get_box_single_anchor3.txt",
  "/home/gouzhihong/project/data/get_box_single_anchor4.txt",
};

char box_name[5][100] = {
  "/home/gouzhihong/project/data/get_box_single_box0.txt",
  "/home/gouzhihong/project/data/get_box_single_box1.txt",
  "/home/gouzhihong/project/data/get_box_single_box2.txt",
  "/home/gouzhihong/project/data/get_box_single_box3.txt",
  "/home/gouzhihong/project/data/get_box_single_box4.txt",
};

const int use_sigmoid_cls = 1; 
const int nms_pre = 1000;
const int nms_post = 1000;
const int max_num = 1000;
const int height = 1199;
const int width = 800;
const int min_bbox_size = 0;

void get_box_single(int idx, int *level, int *n, int *c, int *h, int *w, float * mlvl_proposals, int *p_sum_level, float * scores){

  ifstream cls_File;
  cls_File.open(cls_name[idx], ios::in);
  ifstream anchor_File;
  anchor_File.open(anchor_name[idx], ios::in);
  ifstream box_File;
  box_File.open(box_name[idx], ios::in);

  float *h_cls;
  int *h_index;
  
  float *h_anchor;
  float *h_box;
  float *h_proposals;
  int *h_inds;
  
  float *d_cls;     
  int *d_index;
  float *d_anchor;
  float *d_box;
  float *d_cls_out; 
  float *d_box_out;
  float *d_score; 
  float *d_anchor_slice;
  float *d_box_slice; 
  float *d_gbox;
  float *d_box1; 
  float *d_proposals;
  
  int num = level[idx];
  int keep = num < nms_pre ? num : nms_pre;
  
  h_cls = (float*)malloc(level[idx] * sizeof(float));
  h_index = (int*)malloc(level[idx] * sizeof(int));
  h_anchor = (float*)malloc(level[idx] * 4 *sizeof(float));
  h_box = (float*)malloc(level[idx] * 4 *sizeof(float));
  h_proposals = (float*)malloc(keep * 5 * sizeof(float));
  h_inds = (int*)malloc(keep * sizeof(int));
  
  for(int i = 0; i < level[idx]; ++i){
    cls_File >> h_cls[i];
    h_index[i] = i;
  }
  cls_File.close(); 
   
  for(int i = 0; i < level[idx]; ++i){
    for(int j =0; j < 4; j++){
      anchor_File >> h_anchor[i*4 + j];
    }
  }
  anchor_File.close();
  for(int i = 0; i < level[idx]; ++i){
    for(int j =0; j < 4; j++){
      box_File >> h_box[i*4 + j];
    }
  }
  box_File.close();
  
  ErrChk(cudaMalloc(&d_cls,  level[idx] * sizeof(float)));
  ErrChk(cudaMalloc(&d_index, level[idx] * sizeof(int)));
  ErrChk(cudaMalloc(&d_anchor, level[idx] * 4* sizeof(float)));
  ErrChk(cudaMalloc(&d_box, level[idx] * 4* sizeof(float)));
  ErrChk(cudaMalloc(&d_cls_out, level[idx] * sizeof(float))); 
  ErrChk(cudaMalloc(&d_box_out, level[idx] * 4 *sizeof(float)));
  ErrChk(cudaMalloc(&d_score, keep * sizeof(float)));
  ErrChk(cudaMalloc(&d_anchor_slice, keep * 4* sizeof(float)));
  ErrChk(cudaMalloc(&d_box_slice, keep * 4* sizeof(float)));
  ErrChk(cudaMalloc(&d_gbox, keep * 4* sizeof(float)));
  ErrChk(cudaMalloc(&d_box1, keep * 4* sizeof(float)));
  ErrChk(cudaMalloc(&d_proposals, keep * 5* sizeof(float)));

  ErrChk(cudaMemcpy(d_cls, h_cls, level[idx] *sizeof(float), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(d_index, h_index, level[idx] *sizeof(int), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(d_box, h_box, level[idx] * 4 *sizeof(float), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(d_anchor, h_anchor, level[idx] * 4 *sizeof(float), cudaMemcpyHostToDevice));
  
  cudnnHandle_t cudnn;
  ErrChk(cudnnCreate(&cudnn));
  cudnnTensorDescriptor_t in_desc, out_desc;
  cudnnTensorTransformDescriptor_t transformDesc;
  ErrChk(cudnnCreateTensorDescriptor(&in_desc));
  ErrChk(cudnnCreateTensorDescriptor(&out_desc));
  ErrChk(cudnnCreateTensorTransformDescriptor(&transformDesc));
  cudnnActivationDescriptor_t activationDesc;
  ErrChk(cudnnCreateActivationDescriptor(&activationDesc)); 

  float alpha = 1.0f;
  float beta = 0.0f;
  float msecTotal = 0.0f;
  float msecPerkernel;

  clock_t start_cpu;
  clock_t end_cpu;
  start_cpu = clock();

  ErrChk(cudnnSetTensorTransformDescriptor(transformDesc, 4, CUDNN_TENSOR_NHWC, NULL, NULL, NULL, CUDNN_TRANSFORM_UNFOLD));
  ErrChk(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n[idx], c[idx], h[idx], w[idx]));   //input nchw
  ErrChk(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n[idx], c[idx], h[idx], w[idx]));  //output nhwc
  ErrChk(cudnnTransformTensorEx(cudnn, transformDesc, &alpha, in_desc, d_cls, &beta, out_desc, d_cls_out));

  
  cudnnTensorDescriptor_t in_desc_1, out_desc_1;
  cudnnTensorTransformDescriptor_t transformDesc_1;
  ErrChk(cudnnCreateTensorDescriptor(&in_desc_1));
  ErrChk(cudnnCreateTensorDescriptor(&out_desc_1));
  ErrChk(cudnnCreateTensorTransformDescriptor(&transformDesc_1));
  ErrChk(cudnnSetTensorTransformDescriptor(transformDesc_1, 4, CUDNN_TENSOR_NHWC, NULL, NULL, NULL, CUDNN_TRANSFORM_UNFOLD));
  ErrChk(cudnnSetTensor4dDescriptor(in_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n[idx], 4*c[idx], h[idx], w[idx]));   //input nchw
  ErrChk(cudnnSetTensor4dDescriptor(out_desc_1, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n[idx], 4*c[idx], h[idx], w[idx]));  //output nhwc
  ErrChk(cudnnTransformTensorEx(cudnn, transformDesc_1, &alpha, in_desc_1, d_box, &beta, out_desc_1, d_box_out));

  
  if(use_sigmoid_cls){

    alpha = -1.0f;
    ErrChk(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    ErrChk(cudnnActivationForward(cudnn, activationDesc, &alpha, out_desc, d_cls_out, &beta, out_desc, d_cls_out));
      
    thrust::device_ptr<float> dev_data_ptr(d_cls_out);
    thrust::device_ptr<int> dev_index_ptr(d_index);
    thrust::device_ptr<float> dev_scores_ptr(d_score);

    auto first = thrust::make_zip_iterator(thrust::make_tuple(dev_data_ptr, dev_index_ptr));
    auto last  = thrust::make_zip_iterator(thrust::make_tuple(dev_data_ptr + level[idx], dev_index_ptr + level[idx]));
    thrust::sort(first, last);

    thrust::transform(dev_data_ptr, dev_data_ptr + keep, dev_scores_ptr, thrust::negate<float>());
   
    d_index = thrust::raw_pointer_cast(dev_index_ptr); 
    d_score = thrust::raw_pointer_cast(dev_scores_ptr);

    int griddim = (keep * 4 + 1023) / 1024;
    slice_2d_1024<<<griddim, 1024>>>(d_anchor_slice, d_anchor, d_index, keep, 4);
    slice_2d_1024<<<griddim, 1024>>>(d_box_slice, d_box_out, d_index, keep, 4); 
    delta2box_1<<<griddim,1024>>>(d_anchor_slice, d_box_slice, d_gbox, keep, 4.135166556742356);
    delta2box_2<<<griddim,1024>>>(d_gbox, d_box1, keep, width, height);
    
    griddim = (keep * 5 + 1023) / 1024;
    box_and_score_1024<<<griddim, 1024>>>(d_proposals, d_box1, d_score, keep);
    ErrChk(cudaMemcpy(h_proposals, d_proposals, keep * 5 * sizeof(float), cudaMemcpyDeviceToHost));
    
    int num_keep = 0;
    int *num_out = &num_keep;
    nms(h_inds, num_out, h_proposals, keep, 5, 0.7);
    
    int min = num_keep < nms_post ? num_keep : nms_post;
    int count = 0;
    for(int i = *p_sum_level; i < *p_sum_level + min; ++i){
      scores[i] = -1.0 * h_proposals[5*h_inds[count]+4];
      mlvl_proposals[5*i] = h_proposals[5*h_inds[count]];
      mlvl_proposals[5*i+1] = h_proposals[5*h_inds[count]+1];
      mlvl_proposals[5*i+2] = h_proposals[5*h_inds[count]+2];
      mlvl_proposals[5*i+3] = h_proposals[5*h_inds[count]+3];
      mlvl_proposals[5*i+4] = h_proposals[5*h_inds[count++]+4];
    }
    *p_sum_level += min;
    end_cpu = clock();
    double time=(double)(end_cpu - start_cpu)/CLOCKS_PER_SEC;
    cout<<"函数的执行时间 is: "<<time*1000<<"ms"<<endl; 
  }
  /* else{
      cudnnTensorDescriptor_t out_desc2;
      ErrChk(cudnnCreateTensorDescriptor(&out_desc2));
      ErrChk(cudnnSetTensor4dDescriptor(out_desc2, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, n, 2, 1, h*w*c/2));
      ErrChk(cudnnSoftmaxForward(cudnn,CUDNN_SOFTMAX_ACCURATE,CUDNN_SOFTMAX_MODE_CHANNEL,&alpha, out_desc2, out_cls, &beta, out_desc2, out_cls))
    } */
  free(h_cls);
  free(h_index);
  free(h_box);
  free(h_anchor);
  free(h_proposals);
  free(h_inds); 
  
  ErrChk(cudaFree(d_cls));
  ErrChk(cudaFree(d_index));    
  ErrChk(cudaFree(d_anchor)); 
  ErrChk(cudaFree(d_box));
  ErrChk(cudaFree(d_cls_out));
  ErrChk(cudaFree(d_box_out));
  ErrChk(cudaFree(d_score));
  ErrChk(cudaFree(d_anchor_slice));
  ErrChk(cudaFree(d_box_slice)); 
  ErrChk(cudaFree(d_gbox)); 
  ErrChk(cudaFree(d_box1)); 
  ErrChk(cudaFree(d_proposals)); 
  
  ErrChk(cudnnDestroyTensorDescriptor(out_desc));
  ErrChk(cudnnDestroyTensorDescriptor(in_desc));
  ErrChk(cudnnDestroyTensorTransformDescriptor(transformDesc));
  ErrChk(cudnnDestroyActivationDescriptor(activationDesc));
  ErrChk(cudnnDestroyTensorDescriptor(out_desc_1));
  ErrChk(cudnnDestroyTensorDescriptor(in_desc_1));
  ErrChk(cudnnDestroyTensorTransformDescriptor(transformDesc_1));
  ErrChk(cudnnDestroy(cudnn)); 
}

void topk_level(float * mlvl_proposals, int num, int sum_level, int * index, float * scores, float * h_proposals){

  clock_t start_cpu;
  clock_t end_cpu;
  start_cpu = clock();

  float * d_score;
  ErrChk(cudaMalloc(&d_score, sum_level * sizeof(float)));
  ErrChk(cudaMemcpy(d_score, scores, sum_level *sizeof(float), cudaMemcpyHostToDevice));
  
  int * d_index;
  ErrChk(cudaMalloc(&d_index, sum_level * sizeof(int)));
  ErrChk(cudaMemcpy(d_index, index, sum_level *sizeof(int), cudaMemcpyHostToDevice));

  thrust::device_ptr<float> dev_data_ptr(d_score);
  thrust::device_ptr<int> dev_index_ptr(d_index);
  
  auto first = thrust::make_zip_iterator(thrust::make_tuple(dev_data_ptr, dev_index_ptr));
  auto last  = thrust::make_zip_iterator(thrust::make_tuple(dev_data_ptr + sum_level, dev_index_ptr + sum_level));
  thrust::sort(first, last);

  d_index = thrust::raw_pointer_cast(dev_index_ptr); 
  
  float * d_mlvl_proposals;
  ErrChk(cudaMalloc(&d_mlvl_proposals, sum_level * 5 * sizeof(float)));
  ErrChk(cudaMemcpy(d_mlvl_proposals, mlvl_proposals, sum_level * 5 *sizeof(float), cudaMemcpyHostToDevice));
  
  float * d_proposals;
  ErrChk(cudaMalloc(&d_proposals, num * 5 * sizeof(float)));

  int griddim = (num * 5 + 1023) / 1024;
  slice_2d_1024_2<<<griddim, 1024>>>(d_proposals, d_mlvl_proposals, d_index, num, 5);
  
  ErrChk(cudaMemcpy(h_proposals, d_proposals, num * 5 * sizeof(float), cudaMemcpyDeviceToHost));
  
  end_cpu = clock();
  double time=(double)(end_cpu - start_cpu)/CLOCKS_PER_SEC;
  cout<<"函数的执行时间 is: "<<time*1000<<"ms"<<endl; 

  ErrChk(cudaFree(d_score)); 
  ErrChk(cudaFree(d_index)); 
  ErrChk(cudaFree(d_mlvl_proposals)); 
  ErrChk(cudaFree(d_proposals)); 
}

int main() {
 
  int num_level = 5;
  int * n = new int[num_level];
  int * c = new int[num_level];
  int * h = new int[num_level];
  int * w = new int[num_level];
  int * level = new int[num_level];
  n[0] = 1; c[0] = 3; h[0] = 200; w[0] = 304; 
  n[1] = 1; c[1] = 3; h[1] = 100; w[1] = 152; 
  n[2] = 1; c[2] = 3; h[2] = 50; w[2] = 76; 
  n[3] = 1; c[3] = 3; h[3] = 25; w[3] = 38; 
  n[4] = 1; c[4] = 3; h[4] = 13; w[4] = 19; 

  level[0] = 182400;
  level[1] = 45600;
  level[2] = 11400;
  level[3] = 2850;
  level[4] = 741;

  float *mlvl_proposals;
  mlvl_proposals = (float*)malloc(nms_pre * num_level * 5 * sizeof(float));
  float *scores;
  scores = (float*)malloc(nms_pre * num_level * sizeof(float));
  
  int sum_level = 0;
  int *p_sum_level = &sum_level;

  for(int idx = 0; idx < num_level; idx++){
    get_box_single(idx, level, n, c, h, w, mlvl_proposals, p_sum_level, scores);
  }
  int *index;
  index = (int*)malloc(sum_level * sizeof(int));
  for(int i = 0 ;i < sum_level; i++){
    index[i] = i;
  }
  int num = max_num < sum_level ? max_num : sum_level;
  float *h_proposals;

  h_proposals = (float*)malloc(num * 5 * sizeof(float));
  topk_level(mlvl_proposals, num, sum_level, index, scores, h_proposals);
  /* for(int i = 0; i < 50; ++i){
    cout<< h_proposals[i] << " ";
    if(i%5==4)
      cout<<endl;
    else
       cout<<" ";
  }  */
  free(scores);
  free(mlvl_proposals);
  free(h_proposals);
  return 0;
}

