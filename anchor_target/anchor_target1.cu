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

#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>

using namespace std;

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

char anchor_name[5][100] = { 
  "/home/gouzhihong/project/data/anchor_target_anchor0.txt",
  "/home/gouzhihong/project/data/anchor_target_anchor1.txt",
  "/home/gouzhihong/project/data/anchor_target_anchor2.txt",
  "/home/gouzhihong/project/data/anchor_target_anchor3.txt",
  "/home/gouzhihong/project/data/anchor_target_anchor4.txt",
};

char flag_name[5][100] ={
  "/home/gouzhihong/project/data/anchor_target_flag0.txt",
  "/home/gouzhihong/project/data/anchor_target_flag1.txt",
  "/home/gouzhihong/project/data/anchor_target_flag2.txt",
  "/home/gouzhihong/project/data/anchor_target_flag3.txt",
  "/home/gouzhihong/project/data/anchor_target_flag4.txt",
};

char box_name[2][100] = {
  "/home/gouzhihong/project/data/anchor_target_box0.txt",
  "/home/gouzhihong/project/data/anchor_target_box1.txt",
};

const int use_sigmoid_cls = 1; 
const int nms_pre = 1000;
const int nms_post = 1000;
const int max_num = 1000;
const int height = 1199;
const int width = 800;
const int min_bbox_size = 0;

typedef unsigned char uint8_t;

// ascii 0: 48 1: 49

extern "C" __global__ void anchor_inside_flags_1024( unsigned char* __restrict__ inside_flags,  unsigned char* __restrict__ flag,
  float* __restrict__ anchor, int num_anchors, int allow, int height, int width) {
  if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < num_anchors) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchors * 4) {
      inside_flags[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((((((((int)flag[((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2))]) == 49) && (-1.0 * allow  <= anchor[((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 2) * 4))])) && (-1.0 * allow  <= anchor[(((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 2) * 4)) + 1)])) && (anchor[(((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 2) * 4)) + 2)] < width + allow)) && (anchor[(((((int)blockIdx.x) * 1024) + ((((int)threadIdx.x) >> 2) * 4)) + 3)] < height + allow)) ? (unsigned char)49 : (unsigned char)48);
    }
  }
}

extern "C" __global__ void box_overlap_1024( float* __restrict__ gt_box,  float* __restrict__ anchor,  float* __restrict__ ious, int num_anchors, int num_gt) {
  float rb[2];
  float lt[2];
  float overlap[1];
  float area[1];
 for (int k = 0; k < 2; ++k) {
   if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchors * num_gt) {
     rb[k] = min(gt_box[((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / num_anchors) * 4) + k) + 2)], anchor[((((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % num_anchors) * 4) + k) + 2)]);
   }
 }
 for (int k1 = 0; k1 < 2; ++k1) {
   if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchors * num_gt) {
     lt[k1] = max(gt_box[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / num_anchors) * 4) + k1)], anchor[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % num_anchors) * 4) + k1)]);
   }
 }
 for (int k2 = 0; k2 < 2; ++k2) {
   if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchors * num_gt) {
     rb[k2] = max(((rb[k2] - lt[k2]) + 1.000000e+00f), 0.000000e+00f);
   }
 }
 if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchors * num_gt) {
   overlap[0] = (rb[0] * rb[1]);
 }
 if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchors * num_gt) {
   area[0] = (((((gt_box[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / num_anchors) * 4) + 2)] - gt_box[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / num_anchors) * 4)]) + 1.000000e+00f) * ((gt_box[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / num_anchors) * 4) + 3)] - gt_box[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) / num_anchors) * 4) + 1)]) + 1.000000e+00f)) + (((anchor[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % num_anchors) * 4) + 2)] - anchor[((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % num_anchors) * 4)]) + 1.000000e+00f) * ((anchor[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % num_anchors) * 4) + 3)] - anchor[(((((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) % num_anchors) * 4) + 1)]) + 1.000000e+00f))) - overlap[0]);
 }
 if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchors * num_gt) {
   ious[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (overlap[0] / area[0]);
 }
}

extern "C" __global__ void argmax_1024( float* __restrict__ A, float * __restrict__ max, unsigned int * __restrict__ max_idx, 
  const int num_gt, const int num_anchor) 
  {
     __shared__ float red_buf0[1024];
     __shared__ unsigned int index_buf0[1024];
     float compute_red_rf[1];
     unsigned int  index[1];
     compute_red_rf[0] = -3.402823e+38f;
     index[0] = 0;
     int start = blockIdx.x * num_anchor;

     for (int k1_outer = 0; k1_outer < (num_anchor + 1023) / 1024; ++k1_outer) {
       if ( k1_outer * 1024 + threadIdx.x < num_anchor) {
           if(compute_red_rf[0] < A[start + k1_outer * 1024 + threadIdx.x]){
               compute_red_rf[0] = A[start + k1_outer * 1024 + threadIdx.x];
               index[0] = k1_outer * 1024 + threadIdx.x;
              }
          }
      }
      __syncthreads();
      ((volatile __shared__ float*)red_buf0)[threadIdx.x] = compute_red_rf[0];
      ((volatile __shared__ unsigned int *)index_buf0)[threadIdx.x] = index[0];
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
      } 
      __syncthreads();
    
      if (((int)threadIdx.x) < 8) {
        if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 8]){
          red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 8];
          index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 8];
        }
      } 
      __syncthreads();
     if (((int)threadIdx.x) < 4) {
       if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 4]){
         red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 4];
         index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 4];
        }
      } 
     __syncthreads();

     if (((int)threadIdx.x) < 2) {
       if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 2]){
         red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 2];
         index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 2];
        }
      } 
     __syncthreads();
     if (((int)threadIdx.x) < 1) {
       if(red_buf0[threadIdx.x] < red_buf0[threadIdx.x + 1]){
         red_buf0[threadIdx.x] = red_buf0[threadIdx.x + 1];
         index_buf0[threadIdx.x] = index_buf0[threadIdx.x + 1];
        }
      } 
     __syncthreads();

     if (((int)threadIdx.x) == 0) {
       max[blockIdx.x] = ((volatile __shared__ float*)red_buf0)[0]; 
       max_idx[blockIdx.x] = ((volatile __shared__ unsigned int*)index_buf0)[0];
      }
}

struct is_ture
{
  __host__ __device__
  bool operator()(uint8_t x)
  {
    return x  == '1';
  }
};

struct is_pos
{
  __host__ __device__
  bool operator()(long x)
  {
    return x > 0;
  }
};

struct is_zero
{
  __host__ __device__
  bool operator()(long x)
  {
    return x == 0;
  }
};

extern "C" __global__ void slice_1d_1024( long * __restrict__ index,  float* __restrict__ overlap, unsigned int *  __restrict__  overlap_index, int num_anchor, float neg_thr, float pos_thr) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchor) {
    if(((0.000000e+00f <= overlap[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) && (overlap[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] < neg_thr))){
      index[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (long)0;
    }
    if(overlap[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] > pos_thr){
      index[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (long)(overlap_index[blockIdx.x * 1024 + threadIdx.x] + 1);
    }
  }
}

extern "C" __global__ void gt_assign( float* __restrict__ overlap,  float* __restrict__ gt_max_overlap,  long * __restrict__ assign_index, int num_anchor) {
  
  int start = blockIdx.x * num_anchor;
  for (int k1_outer = 0; k1_outer < (num_anchor + 1023) / 1024; ++k1_outer) {
    if ( k1_outer * 1024 + threadIdx.x < num_anchor) {
        if(overlap[start + k1_outer * 1024 + threadIdx.x] == gt_max_overlap[blockIdx.x]){
          assign_index[k1_outer * 1024 + threadIdx.x] = (long)(blockIdx.x + 1);
           }
       }
   }
}


/* extern "C" __global__ void box_sample( long* __restrict__ pos_index, long* __restrict__ gt_index, int num_anchor) {
  
  __shared__ int pos_count;
  pos_count = 0;
  for (int k1_outer = 0; k1_outer < (num_anchor + 1023) / 1024; ++k1_outer) {
    int index = k1_outer * 1024 + threadIdx.x;
    if ( index < num_anchor) {
      if(gt_index[index] > 0){
        atomicAdd(&pos_count, 1);
        //atomicExch((pos_index + (*pos_index) - 1), (int)index);
        pos_index[pos_count - 1] = index;
      }
    }
  }
}
 */
void anchor_target_single(float *h_flat_anchors, uint8_t *h_valid_flags, float *h_gt_bboxes, int gt_bboxes_ignore, int gt_labels, float *h_means, float *h_stds, int label_channels, 
  int sampling, int unmap_outputs, int num_anchors, int num_gt, int height, int width, int allowed_border, float neg_thr, float pos_thr, int num_expected_pos){
  float * d_flat_anchors;
  uint8_t * d_valid_flags;
  float * d_gt_bboxes;
  float * d_anchors;

  ErrChk(cudaMalloc(&d_flat_anchors,  num_anchors * 4 * sizeof(float)));
  ErrChk(cudaMalloc(&d_valid_flags,  num_anchors * sizeof(uint8_t)));
  ErrChk(cudaMalloc(&d_gt_bboxes,  num_gt * 4 * sizeof(float)));
  

  ErrChk(cudaMemcpy(d_flat_anchors, h_flat_anchors, num_anchors *4 *sizeof(float), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(d_valid_flags, h_valid_flags, num_anchors *sizeof(uint8_t), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(d_gt_bboxes, h_gt_bboxes, num_gt *4 *sizeof(float), cudaMemcpyHostToDevice));

  uint8_t * d_inside;
  ErrChk(cudaMalloc(&d_inside,  num_anchors * 4 * sizeof(uint8_t)));
  dim3 griddim = (num_anchors * 4 + 1023) / 1024;
  dim3 blockdim = 1024;
  cudnnHandle_t cudnn;
  ErrChk(cudnnCreate(&cudnn));
  cudnnReduceTensorDescriptor_t reduce;
  ErrChk(cudnnCreateReduceTensorDescriptor(&reduce));
  
  ErrChk(cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, CUDNN_32BIT_INDICES));
  cudnnTensorDescriptor_t in_desc, out_desc;
  ErrChk(cudnnCreateTensorDescriptor(&in_desc));
  ErrChk(cudnnCreateTensorDescriptor(&out_desc));
  cudnnTensorDescriptor_t in_desc_1, out_desc_1;
  ErrChk(cudnnCreateTensorDescriptor(&in_desc_1));
  ErrChk(cudnnCreateTensorDescriptor(&out_desc_1)); 

  clock_t start_cpu;
  clock_t end_cpu;
  start_cpu = clock();

  /* cudaEvent_t start; 
  cudaEventCreate(&start);
  cudaEvent_t stop;
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL); */

  anchor_inside_flags_1024<<<griddim, blockdim>>>(d_inside, d_valid_flags, d_flat_anchors, num_anchors, allowed_border, height, width);
  
  thrust::device_ptr<float> dev_anchor_ptr(d_flat_anchors);
  thrust::device_ptr<uint8_t> dev_flag_ptr(d_inside);

  int num_valid_anchors;
  num_valid_anchors = thrust::count_if(dev_flag_ptr, dev_flag_ptr + num_anchors * 4, is_ture());
  num_valid_anchors /= 4;
  cout << num_valid_anchors << endl;

  ErrChk(cudaMalloc(&d_anchors,  num_valid_anchors * 4 * sizeof(float)));
  thrust::device_ptr<float> d_anchors_ptr(d_anchors);
  
  thrust::copy_if(dev_anchor_ptr, dev_anchor_ptr + num_anchors * 4, dev_flag_ptr, d_anchors_ptr, is_ture()); 
  d_anchors = thrust::raw_pointer_cast(d_anchors_ptr); 
  
  if(sampling){
    if(num_gt == 0 || num_valid_anchors == 0){
      exit(1);
    }
    float * d_overlaps;
    ErrChk(cudaMalloc(&d_overlaps,  num_valid_anchors* num_gt * sizeof(float)));
    box_overlap_1024<<<(num_valid_anchors*num_gt+1023)/1024, 1024>>>(d_gt_bboxes, d_anchors, d_overlaps, num_valid_anchors, num_gt);
    
    long * d_assigned_gt_inds;
    ErrChk(cudaMalloc(&d_assigned_gt_inds, num_valid_anchors * sizeof(long)));
    
    thrust::device_ptr<long> dev_assign_ptr(d_assigned_gt_inds);
    thrust::fill(dev_assign_ptr, dev_assign_ptr + num_valid_anchors, -1);
    
    unsigned int * d_argmax_overlaps;
    size_t size = num_valid_anchors * sizeof(unsigned int);
    ErrChk(cudaMalloc(&d_argmax_overlaps, size));
    float * d_max_overlaps;
    ErrChk(cudaMalloc(&d_max_overlaps, num_valid_anchors * sizeof(float)));
    
    unsigned int * d_gt_argmax_overlaps;
    size_t size2 = num_gt * sizeof(unsigned int);
    ErrChk(cudaMalloc(&d_gt_argmax_overlaps, size2));
    float * d_gt_max_overlaps;
    ErrChk(cudaMalloc(&d_gt_max_overlaps, num_gt * sizeof(float)));

    unsigned int * d_workspace;
    size_t workspace_size = 2 * num_valid_anchors * num_gt *sizeof(float);
    ErrChk(cudaMalloc(&d_workspace, workspace_size));
    
    float alpha = 1.0;
    float beta = 0.0;
  
    ErrChk(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, num_gt, num_valid_anchors));   //input nchw
    ErrChk(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, num_valid_anchors));  //output nhwc
    ErrChk(cudnnReduceTensor(cudnn, reduce, d_argmax_overlaps, size, d_workspace, workspace_size, &alpha, in_desc, d_overlaps, &beta, out_desc, d_max_overlaps)); 
    
    argmax_1024<<<num_gt, 1024>>>(d_overlaps, d_gt_max_overlaps, d_gt_argmax_overlaps, num_gt, num_valid_anchors);
    slice_1d_1024<<<(num_valid_anchors + 1023)/1024, 1024>>>(d_assigned_gt_inds, d_max_overlaps, d_argmax_overlaps, num_valid_anchors, neg_thr, pos_thr);
    gt_assign<<<1, 1024>>>(d_overlaps, d_gt_max_overlaps, d_assigned_gt_inds, num_valid_anchors);
    /* ErrChk(cudnnSetTensor4dDescriptor(in_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, num_gt, num_valid_anchors));   //input nchw
    ErrChk(cudnnSetTensor4dDescriptor(out_desc_1, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, num_gt, 1));  //output nhwc
    ErrChk(cudnnReduceTensor(cudnn, reduce, d_gt_argmax_overlaps, size2, d_workspace, workspace_size, &alpha, in_desc_1, d_overlaps, &beta, out_desc_1, d_gt_max_overlaps));  */
    
    //thrust::device_ptr<long> dev_anchor_ptr(d_assigned_gt_inds);

    int num_pos = thrust::count_if(dev_assign_ptr, dev_assign_ptr + num_valid_anchors, is_pos());
    int num_neg = thrust::count_if(dev_assign_ptr, dev_assign_ptr + num_valid_anchors, is_zero());

   /*  long * d_pos_inds;
    ErrChk(cudaMalloc(&d_pos_inds, num_pos * sizeof(long)));
    box_sample<<<(num_valid_anchors + 1023)/1024, 1024>>>(d_pos_inds, d_assigned_gt_inds, num_valid_anchors); */

    cudaDeviceSynchronize();

    end_cpu = clock();
    double time=(double)(end_cpu - start_cpu)/CLOCKS_PER_SEC;
    cout<<"函数的执行时间 is: "<<time*1000<<"ms"<<endl; 
    cout << num_pos << " " << num_neg << endl;

    long * h_assigned_gt_inds;
    h_assigned_gt_inds = (long*)malloc(num_valid_anchors * sizeof(long));
    ErrChk(cudaMemcpy(h_assigned_gt_inds, d_assigned_gt_inds, num_valid_anchors * sizeof(long), cudaMemcpyDeviceToHost));

    /* long * h_pos_inds;
    h_pos_inds = (long*)malloc(num_pos * sizeof(long));
    ErrChk(cudaMemcpy(h_pos_inds, d_pos_inds, num_pos * sizeof(long), cudaMemcpyDeviceToHost));
    for(int i = 0; i < num_pos; i++)
    cout<<h_pos_inds[i] << endl; */

    ErrChk(cudnnDestroyTensorDescriptor(out_desc));
    ErrChk(cudnnDestroyTensorDescriptor(in_desc));
    ErrChk(cudnnDestroyTensorDescriptor(out_desc_1));
    ErrChk(cudnnDestroyTensorDescriptor(in_desc_1));
    ErrChk(cudnnDestroy(cudnn)); 

    long sum = 0;
    for(int i = 0; i < num_valid_anchors; ++i){
      sum += h_assigned_gt_inds[i];
      /* cout<< fixed << setprecision(7) << h_overlaps[i] << " ";
      if(i % 4 == 3){
        cout << endl;
      }
      else{
        cout << " ";
      }  */
    }
    cout << sum << endl;
  }
  else{
    ;
  }
  
  
  
  //ErrChk(cudaMemcpy(h_inside, d_inside, num_anchors * 4 * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  /* int sum = 0;
  for(int i = 0; i < num_anchors * 4; i++){
    if(h_inside[i] == '1')
    sum++;
  }
  cout << sum << endl; */

}

int main() {
 
  int num_level = 5;
  int * level = new int[num_level];

  level[0] = 201600;
  level[1] = 50400;
  level[2] = 12600;
  level[3] = 3150;
  level[4] = 819;
  
  int num_anchors = 268569;
  int num_gt = 4;

  float * h_flat_anchors;
  uint8_t * h_valid_flags;
  float * h_gt_bboxes;
  float * h_stds;
  float * h_means;

  h_flat_anchors = (float*)malloc(num_anchors * 4 * sizeof(float));
  h_valid_flags = (uint8_t*)malloc(num_anchors * sizeof(uint8_t));
  h_gt_bboxes = (float*)malloc(num_gt * 4 * sizeof(float));
  h_stds = (float*)malloc(4 * sizeof(float));
  h_means = (float*)malloc(4 * sizeof(float));

  h_stds[0] = 1.0; h_stds[1] = 1.0; h_stds[2] = 1.0; h_stds[3] = 1.0;
  h_means[0] = 0.0; h_means[1] = 0.0; h_means[2] = 0.0; h_means[3] = 0.0;

  int start = 0;
  for(int idx = 0; idx < num_level; idx++){
    ifstream anchor_File;
    anchor_File.open(anchor_name[idx], ios::in);
    ifstream flag_File;
    flag_File.open(flag_name[idx], ios::in);

    for(int i = 0; i < level[idx]; ++i){
      for(int j =0; j < 4; j++){
        anchor_File >> h_flat_anchors[(i + start)*4 + j];
      } 
    }
    anchor_File.close(); 
     
    for(int i = 0; i < level[idx]; ++i){
      flag_File >>  h_valid_flags[i + start];
    }
    flag_File.close();
    start += level[idx];
  }
  
  ifstream box_File;
  box_File.open(box_name[0], ios::in);
  for(int i = 0; i < num_gt; ++i){
    for(int j =0; j < 4; j++){
      box_File >> h_gt_bboxes[i*4 + j];
    }
  }
  box_File.close();
  
  int gt_bboxes_ignore = 0;
  int gt_labels = 0;
  int label_channels = 1;
  int sampling = 1;
  int unmap_outputs = 1;
  int shape1 = 750;
  int shape2 = 1333;
  int allowed_border = 0;
  float neg_thr = 0.3;
  float pos_thr = 0.7;
  int num_expected_pos = 128;
  anchor_target_single(h_flat_anchors, h_valid_flags, h_gt_bboxes, gt_bboxes_ignore, gt_labels, h_means, h_stds, label_channels, 
    sampling, unmap_outputs, num_anchors, num_gt, shape1, shape2, allowed_border, neg_thr, pos_thr, num_expected_pos);
  
  /* for(int i = num_anchors * 4; i > num_anchors * 4 - 50; --i){
    cout<< h_flat_anchors[i] << " ";
    if(i%5==4)
      cout<<endl;
    else
       cout<<" ";
  }
  for(int i = num_anchors; i > num_anchors - 50; --i){
    printf("%c ", h_valid_flags[i]);
    cout<< h_valid_flags[i] << " ";
    if(i%5==4)
      cout<<endl;
    else
       cout<<" ";
  }  */
  /* int sum = 0;

  for(int i = 0; i < num_anchors; i++){
    if(h_valid_flags[i] =='1')
    sum++;
  }
  cout << sum << endl; */
 /*  for(int i = 0; i < 16; ++i){
    cout<< h_gt_bboxes[i] << " ";
    if(i%4==3)
      cout<<endl;
    else
       cout<<" ";
  }  */
  return 0;
}


