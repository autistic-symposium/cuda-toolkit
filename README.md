## CUDA for Deep Learning 

<br>

- 👉 **Parallel computing platform and API by NVIDIA.**
- 👉 **Allows to use CUDA-enabled GPU for general purpose processing.**


<br>

---

### GPU Cards

<br>

- When discussing CUDA there are four designations:
    - model name/number of the card.
    - model name/number of the GPU.
    - compute capability.
    - architecture class:
        - Tesla (2007)
        - Fermi (2010)
        - Kepler (2012)
        - Maxwell (2014)
        - Pascal (2016)

- Maxwell TitanX card is about six teraops per second.
    - quoted DRAM bandwidth is 250 GB/s.

- Pascal is about double that number, with between 10-11 teraops per second.

- To do message passing, we can go from the SRAM of one chip to the SRAM of another; there’s no complicated memory hierarchy.

- With deep learning, you know all the steps of what you’re going to do up front (as soon as network is defined) you know what those back propagation steps look like.

- You can verify your card's computing capability at [cuda-gpus page](https://developer.nvidia.com/cuda-gpus).

- Check PCIe communication bus that connects between CPU and peripheral devices such as graphic card:

```
$ lspci | grep -i nvidia
```

<br>

---

### CUDA Essentials

<br>

#### Architecture

- CUDA employs the SIMT (single instruction multiple thread) model.

- CUDA GPUs contain numerous fundamental computing units called cores:
    - each core includes an arithmetic logic unit (ALU) and a floating point unit (FPU).
    - cores are collected into groups called streaming multiprocessors (SMs).

- computing tasks are break into subtasks called threads and organized into blocks:
    - blocks are divided into warps whose size matches the number of cores in an SM.

    - each warp gets assigned to a particular SM for execution.
    
    - the SM’s control unit directs each of its cores to execute the same instruction simultaneously for each thread in the assigned warp.
        - single instruction multiple thread

<br>

#### Dynamic memory management

- Stack memory is limited, and you will produce segmentation fault errors for a N very large or not known at compile time:
    - the header ```stdlib.h``` supports calling dynamic memory management functions.
    -  ```float *in = (float*)calloc(N, sizeof(float))``` —> heap memory, N can be a large array size.
        - and then ```free(in)```.

- The time required to access data increases with the distance between the computing core and the memory location where the data is stored.

- The time that a core spends waiting for data is called latency, and a CPU is designed to minimize latency by dedicating lots of space to storing data where it can be accessed quickly:
    
    - the allocation of space on the GPU is very different, most of the chip area is dedicated to lots of computing cores organized into SMs with a shared control unit. Instead of minimizing latency (which would require a significant amount of cache per core), the GPU aims to hide latency.
    
    - when execution of a warp requires data that is not available, the SM switches to executing another warp whose data is available
     
    - the focus is an overall computing throughput rather than on the speed of an individual core

- The essential software construct is a  function called *kernel* that spawns a large collection of computational threads organized into groups that can be assigned to SMs.
    - We launch a kernel to create a computational grid composed of blocks of threads.

- CUDA provides each thread with built-in index variables, which replace the loop index in serial code.

<br>

#### Profiling your code to find bottlenek

You only benefit from massive parallelism of GPUs when using batch computations
large batch sizes and CNNs.

```
python -m cProfile -s time your_script.py > results.txt
```

<br>

#### Basic Workflow

- The basic tasks needed for CUDA parallelism:
    - launching a kernel with specified grid dimensions (number of blocks and threads).
    - specifying that functions should be compiled to run on the device or host.
    - allocating memory and transferring data.

- Workflow:
    - copy input data to the device once and leave it there.
    - launch a kernel that does a significant amount of work.
    - copy results back to host only once.

- Or a streamlined alternative called unified memory, where you can have an array that is accessible from both host and device:
    - create a single managed array that can be accessed from both host and device.
    - CUDA system schedules and executes the transfers.
    - cudaMallocManaged().

<br>

#### Qualifiers

<br>

- __global__ kernel, called from host, executed on device:
        - cannot return value (always void).
        - kernels provide dimension and index variables for each block and thread:
            - dimensions: gridDim specifies the number of blocks in the grid, blockDim specifies the number of threads in each block.
            - index variables: blockIdx gives the index of the block in the grid, threadIdx gives the index of the thread within the block.
    - __host__ called from the host executed on host.
    - __device__ called from device and execute on device.

- to indicate the parallel nature and to specify the dimensions of the computational grid, the grid dimensions and block dimensions (inside triple angle brackets or triple chevrons):

    - Name<<<Dg, Db>>(args)
        - Dg is the number of blocks in the grid.
        - Db is the number of threads in a block.

    - They specify the dimensions of the kernel launch.

<br>

---

#### The CUDA runtime API

<br>

- The basic memory calls:
    - cudaMalloc() to allocate device memory.
    - cudaMemcpy() transfer data.
        - copy between host and device across PCIe bus.
    - cudaFree() frees device memory.

- Kernels enable multiple computations to occur in parallel but they also give up control of the order of execution:
    - CUDA provides functions to synchronize and coordinate execution
        - __syncThreads() synchronizes threads within a block
        - cudaDeviceSynchronize() effectively synchronizes all threads in a grid
        - atomic operations, such as atomicAdd(), prevent conflicts associated with multiple threads concurrently accessing a variable
    - CUDA extends the standard C data types to vector of length up to 4, individual components are accessed with the suffixes .x, .y, .z, .w
    - CUDA user the vector type uint3 for the index variable, blockIdx and threadIdx.
        - a uint3 variable is a vector with 3 unsigned integer components.
    - CUDA uses the vector type dim3 for the dimension variables, grimDim and blockDim
        - the dim3 type is equivalent to uint3 with unspecified entries set to 1.

- Replace serial passes through a loop with a computational grid of threads that can execute together.

- Choosing the specific execution configuration that will produce the best performance on a given system is hard, choosing the number of threads in a block to be some multiple of 32 is reasonable since that matches with the number of cuda cores in a SM.

- To avoid issues associated with GPU/CPU synchronization and to provide finer resolution, CUDA includes its own timing mechanisms.
    - cudaEvent_t data type
    - cudaEventCreate() and cudaEventDestroy() for creating and destroying events.
    - cudaEventRecord() for recording time.
    - cudaEventSynchronize() for ensuring completion of asynchronous functions.
    - cudaEventElapsedTime(0 for converting a pair of event records to elapsed time.

<br>

---

#### Building Apps

<br>

##### NVCC

<br>

```
$ nvcc -g -G -Xcompiler -Wall main.cpp
```
-g debug
-G device debug
-Xcompiler compiler options
-Wall all the warning messages

<br>

##### cuda-gdb

<br>

- debugging:
    - compile with flags -g -G

```
$ cuda-gdb <binary>
> start
> info locals
> next
> continue
> quit
> break 12
> run
> cuda kernel block thread
```

<br>

##### nvprof

<br>

```
$ nvprof --print-gpu-trace ./main --benchmark
$ nvprof --analysis-metrics -o  nbody-analysis.nvprof ./main --benchmark -numdevices=2 -i=1
```

<br>

##### cuda-memcheck

<br>

Runtime error detector tool that detect memory leaks, memory access errors, and hardware errors:

```
$ cuda-memcheck <exe>
```

<br>

#### PTX and SASS Assembly Debugging

<br>

- PTX is a low-level parallel-thread execution virtual machine and instruction set architecture (ISA). PTX exposes the GPU as a parallel computing device.

- SASS is the low-level assembly language that compiles to binary microcode, which executes natively on NVIDIA GPU hardware.

<br>

#### Texture Memory

<br>

- Although NVIDIA designed the texture units for the classical OpenGL and DirectX rendering pipelines, texture memory has some properties that make it extremely useful for computing.

- Texture memory is cached on chip, so in some situations it will provide higher effective bandwidth by reducing memory requests to off-chip DRAM.

- Texture caches are designed for graphics applications where memory access patterns exhibit a great deal of spatial locality. In a computing application, this roughly implies that a thread is likely to read from an address “near” the address that nearby threads read.

- The texture cache is optimized for 2D spatial locality, so threads of the same warp that read texture addresses that are close together will achieve best performance.

- The process of reading a texture is called a texture fetch. The first parameter of a texture fetch specifies an object called a texture reference.

<br>

---

## Index & Architecture

<br>

### 2D Grids indices

<br>

```
int row = blockIdx.x * blockDim.x + threadIdx.x
int col = blockIdx.y * blockDim.y + threadIdx.y
```

and a general flat index can be written with

```
i = r*w + c
```

<br>

### 2D Block size

<br>

```
dim blockSize(TX, TY)

int bx = (W + blockSize.x - 1)/blockSize.x
int by = (H + blockSize.y - 1)/blocksSize.y

dim3 gridSize = dim3(bx, by)
```

And then you can launch:

```
KernelName<<<gridSize, blockSize>>> (args)
```

<br>

---

## Getting started with CUDA for Deep Learning

<br>

Check the directories under `0-6`.

<br>

---

## Some Useful References

<br>

* **[CUDA C Best Practices Guide](http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)**
* **[CUDA for Engineers](https://www.amazon.com/CUDA-Engineers-Introduction-High-Performance-Computing/dp/013417741X)**
