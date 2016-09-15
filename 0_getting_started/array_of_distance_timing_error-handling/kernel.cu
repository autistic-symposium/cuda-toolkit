#include "kernel.h"
#include <stdio.h>

#define M 100 // number of times to do the data transfer
#define TPB 32


__device__ float distance(float x1, float x2)
{
	return sqrt((x2 - x1) *(x2 - x1));
}

__global__ void distanceKernel(float *d_out, float *d_in, float ref)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	const float x = d_in[i];
	d_out[i] = distance(x, ref);
	printf("i = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);
}

// wrapper or launcher
void distanceArray(float *out, float *in, float ref, int len)
{
	// Create event variables for timing
	cudaEvent_t startMemcpy, stopMemcpy;
	cudaEvent_t startKernel, stopKernel;
	cudaEventCreate(&startMemcpy);
	cudaEventCreate(&stopMemcpy);
	cudaEventCreate(&startKernel);
	cudaEventCreate(&stopKernel);

	float *d_in = 0;
	float *d_out = 0;

	cudaError_t err = cudaMalloc(&d_in, len*sizeof(float));
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

	err = cudaMalloc(&d_out, len*sizeof(float));
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

	cudaEventRecord(startMemcpy);
	// copy input data from host to device M times
	for (int i = 0; i < M; ++i)
	{
		cudaMemcpy(d_in, in, len*sizeof(float), cudaMemcpyHostToDevice);
	}

	cudaEventRecord(stopMemcpy);

	cudaEventRecord(startKernel);
	distanceKernel<<<len/TPB, TPB>>>(d_out, d_in, ref);
	cudaEventRecord(stopKernel);
	cudaError_t errSync = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();

	// info on kernel launch errors
	if (errSync != cudaSuccess) printf("%s\n", cudaGetErrorString(errSync));

	// info on kernel execution errors
	if (errAsync != cudaSuccess) printf("%s\n", cudaGetErrorString(errAsync));

	err= cudaMemcpy(out, d_out, len*sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));


	// Ensure timed events have stopped.
	cudaEventSynchronize(stopMemcpy);
	cudaEventSynchronize(stopKernel);

	// Convert event records to time and output
	float memcpyTimeInMs = 0;
	cudaEventElapsedTime(&memcpyTimeInMs, startMemcpy, stopMemcpy);
	float kernelTimeInMs = 0;
	cudaEventElapsedTime(&kernelTimeInMs, startKernel, stopKernel);	

	printf("kernel time (ms): %f\n", kernelTimeInMs);
	printf("data transfer time (ms): %f\n", memcpyTimeInMs);	

	cudaFree(d_in);
	cudaFree(d_out);

}