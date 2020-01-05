#NVCC_FLAGS= -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 
NVCC_FLAGS= -gencode arch=compute_61,code=sm_61
anchor_target.out:anchor_target.cu Makefile
	nvcc $< -o $@ -O3 -w -lcudnn ${NVCC_FLAGS}
clean:
	rm -rf anchor_target.out 
