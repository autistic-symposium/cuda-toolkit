NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall 

main: kernel.cu
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ 