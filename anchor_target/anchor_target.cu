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
#include <algorithm>
#include <curand.h>
#include <curand_kernel.h>
#include <map>

#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>
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

struct not_neg
{
  __host__ __device__
  bool operator()(long x)
  {
    return x >= 0;
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

extern "C" __global__ void generate_pos_index(long * __restrict__ pos_index,  long* __restrict__ index, int num_anchor) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchor) {
      pos_index[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (((long)0 < index[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) ? (long)((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) : (long)0);
    }
}

extern "C" __global__ void generate_neg_index(long * __restrict__ neg_index,  long* __restrict__ index, int num_anchor) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchor) {
      neg_index[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (((long)0 == index[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) ? (long)((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) : (long)(-1));
    }
}

extern "C" __global__ void rand_sample( long* __restrict__ neg_index,  long* __restrict__ index,  long* __restrict__ rand_index, int num_neg) {
    if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_neg) {
      neg_index[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = index[rand_index[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]];
    }
}

extern "C" __global__ void set_pos_box_32( float* __restrict__ pos_boxes, float* __restrict__ pos_gt_boxes, float* __restrict__ boxes, 
    float* __restrict__ gt_boxes,  long* __restrict__ pos_inds, long * gt_inds, int num_pos) {
    if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < num_pos) {
      if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_pos * 4) {
        pos_boxes[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = boxes[((pos_inds[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2))] * (long)4) + ((long)(((int)threadIdx.x) & 3)))];
        pos_gt_boxes[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = gt_boxes[(((gt_inds[pos_inds[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2))]] * (long)4) + ((long)(((int)threadIdx.x) & 3))) - (long)4)];
      }
    }
}

extern "C" __global__ void box2delta_32(float* __restrict__ proposal, float* __restrict__ gt, float* __restrict__ stds,  float* __restrict__ mean, float* __restrict__ deltas, int num_pos) {
    float gbox[1];
    float pbox[3];
   if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < num_pos) {
     gbox[0] = (((((int)threadIdx.x) & 3) < 2) ? ((gt[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + gt[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) + 2)]) * 5.000000e-01f) : ((gt[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] - gt[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) - 2)]) + 1.000000e+00f));
   }
   for (int j = 0; j < 3; ++j) {
     if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < num_pos) {
       if ((j + (((int)threadIdx.x) & 3)) < 4) {
         pbox[j] = (((j + (((int)threadIdx.x) & 3)) < 2) ? ((proposal[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) + j)] + proposal[((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) + j) + 2)]) * 5.000000e-01f) : ((proposal[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) + j)] - proposal[((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) + j) - 2)]) + 1.000000e+00f));
       }
     }
   }
   if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < num_pos) {
     gbox[0] = (((((int)threadIdx.x) & 3) < 2) ? ((gbox[0] - pbox[0]) / pbox[2]) : __logf((gbox[0] / pbox[0])));
   }
   if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < num_pos) {
     if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_pos * 4) {
       deltas[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = ((gbox[0] / stds[(((int)threadIdx.x) & 3)]) + mean[(((int)threadIdx.x) & 3)]);
     }
   }
}

extern "C" __global__ void set_zero_1d_1024( long* __restrict__ array1,  float* __restrict__ array2, int dim1) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < dim1) {
        array1[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = (long)0;
        array2[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
    }  
}

extern "C" __global__ void set_pos_box_weight_32( float* __restrict__ box_target, float* __restrict__ box_weight,  float* __restrict__ pos_box_target,  long* __restrict__ pos_inds, int num_pos) {
    if (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2)) < num_pos) {
        if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_pos *4) {
            box_target[((pos_inds[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2))] * (long)4) + ((long)(((int)threadIdx.x) & 3)))] = pos_box_target[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))];
            box_weight[((pos_inds[((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 2))] * (long)4) + ((long)(((int)threadIdx.x) & 3)))] = 1.0;
        }
    }
}

extern "C" __global__ void set_zero_2d_1024( float* __restrict__ array1,  float* __restrict__ array2, int dim1, int dim2) {
    if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < dim1) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < dim1 * dim2) {
        array1[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
        array2[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = 0.000000e+00f;
      }
    }
}

extern "C" __global__ void set_pos_label_weight_32( long* __restrict__ label, float* __restrict__ label_weight, long* __restrict__ pos_inds, int num_pos, float pos_weight) {
    if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_pos) {
        label[pos_inds[blockIdx.x * 32 + threadIdx.x]] = (long)1;
        label_weight[pos_inds[blockIdx.x * 32 + threadIdx.x]] = pos_weight;
    } 
}

extern "C" __global__ void set_neg_label_weight_32(float* __restrict__ label_weight, long* __restrict__ neg_inds, int num_neg, float neg_weight) {
    if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < num_neg) {
        label_weight[neg_inds[blockIdx.x * 32 + threadIdx.x]] = neg_weight;
    } 
}

extern "C" __global__ void unmap_box( float* __restrict__ box_targets, float* __restrict__ box_weights, unsigned char * __restrict__ inside_flags, long* __restrict__ inside_flags4, float* __restrict__ box_target, float* __restrict__ box_weight, int num_ele) {
    if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_ele) {
      box_targets[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((48 < ((int)inside_flags[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))])) ? box_target[(((int)inside_flags4[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) - 1)] : 0.000000e+00f);
      box_weights[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((48 < ((int)inside_flags[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))])) ? box_weight[(((int)inside_flags4[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) - 1)] : 0.000000e+00f);
    }
}

extern "C" __global__ void unmap_label( long* __restrict__ labels, float* __restrict__ label_weights, unsigned char * __restrict__ inside_flags, long* __restrict__ inside_flags4, long* __restrict__ label, float* __restrict__ label_weight, int num_ele) {
  if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_ele) {
    labels[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((48 < ((int)inside_flags[(blockIdx.x * 1024 + threadIdx.x) * 4])) ? label[(((int)inside_flags4[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) - 1)] : (long) 0);
    label_weights[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((48 < ((int)inside_flags[(blockIdx.x * 1024 + threadIdx.x) * 4])) ? label_weight[(((int)inside_flags4[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) - 1)] : 0.000000e+00f);
  }
}

extern "C" __global__ void copy_flag( long* __restrict__ inside_flag4, long* __restrict__ inside_flag1, unsigned char* __restrict__ inside_flags, int num_anchors) {
    if (((((int)blockIdx.x) * 256) + (((int)threadIdx.x) >> 2)) < num_anchors) {
      if (((((int)blockIdx.x) * 1024) + ((int)threadIdx.x)) < num_anchors *4) {
        inside_flag4[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))] = ((((int)inside_flags[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) == 49) ? 1 : 0);
        if(threadIdx.x % 4 ==0){
          inside_flag1[(blockIdx.x * 1024 + threadIdx.x)/4] = ((((int)inside_flags[((((int)blockIdx.x) * 1024) + ((int)threadIdx.x))]) == 49) ? 1 : 0);
        }
      }
    }
}

void anchor_target_single(float *h_flat_anchors, uint8_t *h_valid_flags, float *h_gt_bboxes, int gt_bboxes_ignore, int gt_labels, float *h_means, float *h_stds, int label_channels, 
  int sampling, int unmap_outputs, int num_anchors, int num_gt, int height, int width, int allowed_border, float neg_thr, float pos_thr, int num_sum, float pos_weight){
  
  float * d_flat_anchors;
  uint8_t * d_valid_flags;
  float * d_gt_bboxes;
  float * d_means;
  float * d_stds;

  ErrChk(cudaMalloc(&d_flat_anchors,  num_anchors * 4 * sizeof(float)));
  ErrChk(cudaMalloc(&d_valid_flags,  num_anchors * sizeof(uint8_t)));
  ErrChk(cudaMalloc(&d_gt_bboxes,  num_gt * 4 * sizeof(float)));
  ErrChk(cudaMalloc(&d_means,  4 * sizeof(float)));
  ErrChk(cudaMalloc(&d_stds,  4 * sizeof(float)));

  ErrChk(cudaMemcpy(d_flat_anchors, h_flat_anchors, num_anchors *4 *sizeof(float), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(d_valid_flags, h_valid_flags, num_anchors *sizeof(uint8_t), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(d_gt_bboxes, h_gt_bboxes, num_gt *4 *sizeof(float), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(d_means, h_means, 4 *sizeof(float), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(d_stds, h_stds, 4 *sizeof(float), cudaMemcpyHostToDevice));

  cudnnHandle_t cudnn;
  ErrChk(cudnnCreate(&cudnn));

  // start form here, cudnn handle should be created in advance
  clock_t start_cpu;
  clock_t end_cpu;
  start_cpu = clock();

  /* float msecTotal = 0.0f;
  float msecPerkernel;
  cudaEvent_t start; 
  cudaEventCreate(&start);
  cudaEvent_t stop;
  cudaEventCreate(&stop);
  cudaEventRecord(start, NULL); */

  cudnnReduceTensorDescriptor_t reduce;
  ErrChk(cudnnCreateReduceTensorDescriptor(&reduce));
  ErrChk(cudnnSetReduceTensorDescriptor(reduce, CUDNN_REDUCE_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES, CUDNN_32BIT_INDICES));
  cudnnTensorDescriptor_t in_desc, out_desc;
  ErrChk(cudnnCreateTensorDescriptor(&in_desc));
  ErrChk(cudnnCreateTensorDescriptor(&out_desc));

  uint8_t * d_inside;
  ErrChk(cudaMalloc(&d_inside, num_anchors * 4 * sizeof(uint8_t)));
  dim3 griddim = (num_anchors * 4 + 1023) / 1024;
  dim3 blockdim = 1024;
  anchor_inside_flags_1024<<<griddim, blockdim>>>(d_inside, d_valid_flags, d_flat_anchors, num_anchors, allowed_border, height, width);
  ErrChk(cudaFree(d_valid_flags));

  thrust::device_ptr<float> dev_anchor_ptr(d_flat_anchors);
  thrust::device_ptr<uint8_t> dev_flag_ptr(d_inside);

  int num_valid_anchors;
  num_valid_anchors = thrust::count_if(dev_flag_ptr, dev_flag_ptr + num_anchors * 4, is_ture());
  num_valid_anchors /= 4;

  float * d_anchors;
  ErrChk(cudaMalloc(&d_anchors,  num_valid_anchors * 4 * sizeof(float)));
  thrust::device_ptr<float> d_anchors_ptr(d_anchors);
  thrust::copy_if(dev_anchor_ptr, dev_anchor_ptr + num_anchors * 4, dev_flag_ptr, d_anchors_ptr, is_ture()); 
  
  ErrChk(cudaFree(d_inside));
  ErrChk(cudaFree(d_flat_anchors));

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
    ErrChk(cudaFree(d_workspace));

    argmax_1024<<<num_gt, 1024>>>(d_overlaps, d_gt_max_overlaps, d_gt_argmax_overlaps, num_gt, num_valid_anchors);
    slice_1d_1024<<<(num_valid_anchors + 1023)/1024, 1024>>>(d_assigned_gt_inds, d_max_overlaps, d_argmax_overlaps, num_valid_anchors, neg_thr, pos_thr);
    gt_assign<<<num_gt, 1024>>>(d_overlaps, d_gt_max_overlaps, d_assigned_gt_inds, num_valid_anchors);
    
    ErrChk(cudaFree(d_overlaps));
    ErrChk(cudaFree(d_argmax_overlaps));
    ErrChk(cudaFree(d_max_overlaps));
    ErrChk(cudaFree(d_gt_argmax_overlaps));
    ErrChk(cudaFree(d_gt_max_overlaps));

    int num_pos = thrust::count_if(dev_assign_ptr, dev_assign_ptr + num_valid_anchors, is_pos());
    int num_neg = thrust::count_if(dev_assign_ptr, dev_assign_ptr + num_valid_anchors, is_zero());

    long * d_sample_index;
    ErrChk(cudaMalloc(&d_sample_index, num_valid_anchors * sizeof(long)));
    long * d_pos_inds;
    ErrChk(cudaMalloc(&d_pos_inds, num_pos * sizeof(long)));

    long * d_neg_temp_inds;
    ErrChk(cudaMalloc(&d_neg_temp_inds, num_neg * sizeof(long)));
    long * d_neg_inds;
    ErrChk(cudaMalloc(&d_neg_inds, (num_sum - num_pos) * sizeof(long)));

    generate_pos_index<<<(num_valid_anchors + 1023)/1024, 1024 >>>(d_sample_index, d_assigned_gt_inds, num_valid_anchors);
    thrust::device_ptr<long> dev_pos_ptr(d_sample_index);
    thrust::copy_if(dev_pos_ptr, dev_pos_ptr + num_valid_anchors, d_pos_inds, is_pos()); 

    generate_neg_index<<<(num_valid_anchors + 1023)/1024, 1024 >>>(d_sample_index, d_assigned_gt_inds, num_valid_anchors);
    thrust::copy_if(dev_pos_ptr, dev_pos_ptr + num_valid_anchors, d_neg_temp_inds, not_neg());

    ErrChk(cudaFree(d_sample_index));
    ErrChk(cudaFree(d_neg_temp_inds));
    
    long *h_rand;
    srand((unsigned long)time(NULL));
    int rand_num = num_sum - num_pos;
    h_rand = (long*)malloc(rand_num * sizeof(long));
    map<long,int> exist;
    for(int i = 0; i < rand_num; i++){
      int sign = 0;
      while(sign == 0){
        long temp = (long) (rand() % num_valid_anchors);
        if(exist.count(temp) == 0){
          h_rand[i] = temp;
          exist.insert(make_pair(temp,1));
          sign = 1;
        }
      }
    }
    random_shuffle(h_rand, h_rand + rand_num);
    ErrChk(cudaMemcpy(d_neg_inds, h_rand, (num_sum - num_pos) * sizeof(long), cudaMemcpyHostToDevice)); 
    
    float *d_pos_box;
    ErrChk(cudaMalloc(&d_pos_box, num_pos*4 *sizeof(float)));
    float *d_pos_gt_box;
    ErrChk(cudaMalloc(&d_pos_gt_box, num_pos*4 *sizeof(float)));
    float *d_deltas;
    ErrChk(cudaMalloc(&d_deltas, num_pos*4 *sizeof(float)));
    
    float *d_box_target;
    ErrChk(cudaMalloc(&d_box_target, num_valid_anchors*4 *sizeof(float)));
    float *d_box_weight;
    ErrChk(cudaMalloc(&d_box_weight, num_valid_anchors*4 *sizeof(float)));
    long *d_label;
    ErrChk(cudaMalloc(&d_label, num_valid_anchors *sizeof(long)));
    float *d_label_weight;
    ErrChk(cudaMalloc(&d_label_weight, num_valid_anchors *sizeof(float)));

    set_pos_box_32<<<(num_pos*4+31)/32,32>>>(d_pos_box, d_pos_gt_box, d_anchors, d_gt_bboxes, d_pos_inds, d_assigned_gt_inds, num_pos);
    box2delta_32<<<(num_pos*4+31)/32,32>>>(d_pos_box, d_pos_gt_box, d_stds, d_means, d_deltas, num_pos);

    ErrChk(cudaFree(d_means));
    ErrChk(cudaFree(d_stds));
    ErrChk(cudaFree(d_gt_bboxes));
    ErrChk(cudaFree(d_anchors));
    ErrChk(cudaFree(d_assigned_gt_inds));
    ErrChk(cudaFree(d_pos_box));
    ErrChk(cudaFree(d_pos_gt_box));

    set_zero_2d_1024<<<(num_valid_anchors*4+1023)/1024,1024>>>(d_box_target, d_box_weight, num_valid_anchors, 4);
    set_pos_box_weight_32<<<(num_pos*4+31)/32,32>>>(d_box_target, d_box_weight, d_deltas, d_pos_inds, num_pos);
    
    ErrChk(cudaFree(d_deltas));

    set_zero_1d_1024<<<(num_valid_anchors+1023)/1024,1024>>>(d_label, d_label_weight, num_valid_anchors);
    set_pos_label_weight_32<<<(num_pos+31)/32,32>>>(d_label, d_label_weight, d_pos_inds, num_pos, pos_weight);
    set_neg_label_weight_32<<<(num_sum-num_pos+31)/32,32>>>(d_label_weight, d_neg_inds, num_sum - num_pos, 1.0);
    
    float *d_box_targets;
    ErrChk(cudaMalloc(&d_box_targets, num_anchors*4 *sizeof(float)));
    float *d_box_weights;
    ErrChk(cudaMalloc(&d_box_weights, num_anchors*4 *sizeof(float)));
    long *d_labels;
    ErrChk(cudaMalloc(&d_labels, num_anchors *sizeof(long)));
    float *d_label_weights;
    ErrChk(cudaMalloc(&d_label_weights, num_anchors *sizeof(float))); 

    long* d_inside4;
    long* d_inside1;
    ErrChk(cudaMalloc(&d_inside4, num_anchors*4 *sizeof(long)));
    ErrChk(cudaMalloc(&d_inside1, num_anchors * sizeof(long)));
    copy_flag<<<(num_anchors*4+1023)/1024,1024>>>(d_inside4, d_inside1, d_inside, num_anchors);

    thrust::device_ptr<long> dev_inside4_ptr(d_inside4);
    thrust::device_ptr<long> dev_inside1_ptr(d_inside1);
    thrust::inclusive_scan(dev_inside1_ptr, dev_inside1_ptr + num_anchors, dev_inside1_ptr); 
    thrust::inclusive_scan(dev_inside4_ptr, dev_inside4_ptr + num_anchors*4, dev_inside4_ptr);
    
    unmap_box<<<(num_anchors*4+1023)/1024,1024>>>(d_box_targets, d_box_weights, d_inside, d_inside4, d_box_target, d_box_weight, num_anchors*4); 

    ErrChk(cudaFree(d_box_target));
    ErrChk(cudaFree(d_box_weight));
    ErrChk(cudaFree(d_inside4));

    unmap_label<<<(num_anchors+1023)/1024,1024>>>(d_labels, d_label_weights, d_inside, d_inside1, d_label, d_label_weight, num_anchors); 
    
    ErrChk(cudaFree(d_label));
    ErrChk(cudaFree(d_label_weight));
    ErrChk(cudaFree(d_inside1));

    cudaDeviceSynchronize();
    /* cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);
    msecPerkernel = msecTotal;
    printf("Time= %.3f msec\n", msecPerkernel);
    */
    end_cpu = clock();
    double time=(double)(end_cpu - start_cpu)/CLOCKS_PER_SEC;
    cout << "cpu runtime: " << time*1000 << "ms" << endl;
    
    long * h_neg_inds;
    h_neg_inds= (long*)malloc(rand_num * sizeof(long));
    ErrChk(cudaMemcpy(h_neg_inds, d_neg_inds, rand_num * sizeof(long), cudaMemcpyDeviceToHost));
    
    /* for(int i = 0; i < rand_num; ++i){
      cout<< h_neg_inds[i] << " ";
      if(i % 10 == 9){
        cout << endl;
      }
      else{
        cout<<" ";
      }  
    }
    cout << endl; */
    ErrChk(cudaFree(d_pos_inds));
    ErrChk(cudaFree(d_neg_inds));
    ErrChk(cudaFree(d_box_targets));
    ErrChk(cudaFree(d_box_weights));
    ErrChk(cudaFree(d_labels));
    ErrChk(cudaFree(d_label_weights));

    ErrChk(cudnnDestroyReduceTensorDescriptor(reduce));
    ErrChk(cudnnDestroyTensorDescriptor(out_desc));
    ErrChk(cudnnDestroyTensorDescriptor(in_desc));
    ErrChk(cudnnDestroy(cudnn));
  }
}

int main() {
 
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

  int num_level = 5;
  int * level = new int[num_level];

  level[0] = 193536;
  level[1] = 48384;
  level[2] = 12096;
  level[3] = 3024;
  level[4] = 756;
  
  int num_anchors = 257796;
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
  int height = 750;
  int width = 1333;
  int allowed_border = 0;
  float neg_thr = 0.3;
  float pos_thr = 0.7;
  int num_sum = 256;
  float pos_weight = 1.0;
  anchor_target_single(h_flat_anchors, h_valid_flags, h_gt_bboxes, gt_bboxes_ignore, gt_labels, h_means, h_stds, label_channels, 
    sampling, unmap_outputs, num_anchors, num_gt, height, width, allowed_border, neg_thr, pos_thr, num_sum, pos_weight);
  cout << "pass" << endl;
  return 0;
}