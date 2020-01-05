template <typename DType>
__global__ void deformable_im2col_gpu_kernel(const int n, const DType* data_im, const DType* data_offset,
  const int height, const int width, const int kernel_h, const int kernel_w,
  const int pad_h, const int pad_w,
  const int stride_h, const int stride_w,
  const int dilation_h, const int dilation_w,
  const int channel_per_deformable_group,  // num_channels
  const int batch_size, const int num_channels, const int deformable_group, // deformable_group = 1
  const int height_col, const int width_col,  // oh ow 
  DType* data_col) {
  CUDA_KERNEL_LOOP(index, n) { //每个线程负责kw*kh个数据
    
    // index index of output matrix
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    //compute deformable group index 暂时没有考虑分组卷积，可以修改，现在只是一组
    const int deformable_group_index = c_im / channel_per_deformable_group; //等于0

    const int h_in = h_col * stride_h - pad_h; // 卷积窗口起始位置
    const int w_in = w_col * stride_w - pad_w;
    DType* data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    // compute deformable group index 计算该线程所负责写入的起始地址 包含9个元素的小竖条
    const DType* data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width; 
    //得到当前图片的当前通道数目，也就是卷积核在与哪张图片的哪个通道作运算，这样偏移才是有效的（hw的一个图片）
    const DType* data_offset_ptr = data_offset + (b_col) * 2 * kernel_h * kernel_w * height_col * width_col;
    // offset, N* kh*kw*2* oH*oW* sizeof(float)); offset起始位置，当前到达第几张图片

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) { //循环写入九个位置
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col; //得到竖直方向的偏移下标
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col; //得到水平方向的偏移下标
        const DType offset_h = data_offset_ptr[data_offset_h_ptr];
        const DType offset_w = data_offset_ptr[data_offset_w_ptr];
        DType val = static_cast<DType>(0); //赋值0
        const DType h_im = h_in + i * dilation_h + offset_h; //空洞和偏移修正得到在当前图片当前通道的实际位置 float
        const DType w_im = w_in + j * dilation_w + offset_w; 
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) { //确保没有越界
          //const DType map_h = i * dilation_h + offset_h;
          //const DType map_w = j * dilation_w + offset_w;
          //const int cur_height = height - h_in;
          //const int cur_width = width - w_in;
          //val = deformable_im2col_bilinear(data_im_ptr, width, cur_height, cur_width, map_h, map_w);
          val = deformable_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im); //双线性插值得到当前图片值 float
        }
        *data_col_ptr = val;
        data_col_ptr += batch_size * height_col * width_col; //进入当前竖直条的下一个位置，data_col增加 nhw,
        // 研究下每一个语句的访存模式吧
        // im_to_col用的更多，几乎是必须的，因为在多数训练过程中，图片都是转化成矩阵进行运算的，而且会把训练数据提前转化过去，在运算之后要转化到nchw的标准形式中去吧
      }
    }
  }
}