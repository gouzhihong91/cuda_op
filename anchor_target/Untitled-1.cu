extern "C" __global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

extern "C" __global__ void rand_shuffle( curandState * state, long* input, int num, int neg_num){
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < num; i += blockDim.x * gridDim.x){
        input[i] = (long)((curand(state + i) + curand(state + num - 1 - i) * 123456) % neg_num);
    }
}

float * h_box_targets;
    h_box_targets = (float*)malloc(num_anchors * 4*sizeof(float));
    ErrChk(cudaMemcpy(h_box_targets, d_box_targets, num_anchors*4 *sizeof(float), cudaMemcpyDeviceToHost));

    float * h_box_weights;
    h_box_weights = (float*)malloc(num_anchors * 4*sizeof(float));
    ErrChk(cudaMemcpy(h_box_weights, d_box_weights, num_anchors*4 *sizeof(float), cudaMemcpyDeviceToHost));
    
    float * h_label_weights;
    h_label_weights = (float*)malloc(num_anchors*sizeof(float));
    ErrChk(cudaMemcpy(h_label_weights, d_label_weights, num_anchors*sizeof(float), cudaMemcpyDeviceToHost));
    
    long * h_labels;
    h_labels = (long*)malloc(num_anchors*sizeof(long));
    ErrChk(cudaMemcpy(h_labels, d_labels, num_anchors*sizeof(long), cudaMemcpyDeviceToHost));

    float sum1, sum2;
    sum1=sum2=0.0;
    for(int i = 0; i < num_anchors*4; i++){
        sum1 += h_box_targets[i];
        sum2 += h_box_weights[i];
    }
    cout<< sum1 << " " << sum2 << endl;
    
    long sum3;
    sum3 =0;
    for(int i = 0; i < num_anchors; i++){
        sum3 += h_labels[i];
    }
    cout<< sum3<< endl;
    

    float sum4;
    sum4 = 0.0;
    for(int i = 0; i < num_anchors; i++){
        sum4 += h_label_weights[i];
    }
    cout<< sum4 << endl;

    long * h_assigned_gt_inds;
    h_assigned_gt_inds = (long*)malloc(num_valid_anchors * sizeof(long));
    ErrChk(cudaMemcpy(h_assigned_gt_inds, d_assigned_gt_inds, num_valid_anchors * sizeof(long), cudaMemcpyDeviceToHost));

     /* long *h_rand;
    h_rand = (long*)malloc(num_neg * sizeof(long));
    for(int i = 0; i < num_neg; i++){
        h_rand[i] = (long)i;
    }
    srand((unsigned long)time(NULL));
    random_shuffle(h_rand, h_rand + num_neg);
    ErrChk(cudaMemcpy(d_rand, h_rand, (num_sum - num_pos) * sizeof(long), cudaMemcpyHostToDevice)); */

    /* ErrChk(cudaMalloc(&d_rand, (num_sum - num_pos) * sizeof(long)));
    thrust::device_ptr<long> dev_rand_ptr(d_rand);
    thrust::sequence(dev_rand_ptr, dev_rand_ptr + num_sum - num_pos); */

    //d_pos_inds = thrust::raw_pointer_cast(dev_pos_ptr2); 
    //rand_sample<<<(num_sum - num_pos + 31)/32, 32>>>(d_neg_inds, d_neg_temp_inds, d_rand, num_sum - num_pos);

    /*
    int rand_num = num_sum - num_pos;
    const int CUDA_NUM_THREADS = 32;
    curandState* devStates;
    cudaMalloc (&devStates, CUDA_NUM_THREADS * sizeof(curandState));
    long *d_rand_sort;
    ErrChk(cudaMalloc(&d_rand_sort, rand_num * sizeof(long)));
    long *h_rand_sort;
    h_rand_sort = (long*)malloc(rand_num * sizeof(long));

    int sign = 1;
    while(sign){
        int seed = (unsigned)time(NULL);
        setup_kernel<<<(rand_num + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(devStates, seed);
        rand_shuffle<<<(rand_num + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS>>>(devStates, d_neg_inds, rand_num, num_neg);

        ErrChk(cudaMemcpy(d_rand_sort, d_neg_inds, (num_sum - num_pos) * sizeof(long), cudaMemcpyDeviceToDevice)); 
        thrust::device_ptr<long> dev_rand_ptr(d_rand_sort);
        thrust::sort(dev_rand_ptr, dev_rand_ptr + rand_num);
        ErrChk(cudaMemcpy(h_rand_sort, d_rand_sort, (num_sum - num_pos) * sizeof(long), cudaMemcpyDeviceToHost));
        int i;
        for(i = 0; i < rand_num - 1; i++){
            if(h_rand_sort[i] == h_rand_sort[i+1]){
                cout << "fail" << endl;
                break;
            }
        }
        if(i == rand_num - 1){
            sign = 0;
        }
    }

    ErrChk(cudaFree(d_rand_sort));
    ErrChk(cudaFree(devStates));
    free(h_rand_sort);
    */