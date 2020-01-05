#include <cublas_v2.h>
#define CUDA_KERNEL_LOOP(i, n)                                                 \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
     i += blockDim.x * gridDim.x)

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