//Based on the work of Andrew Krepps
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

__global__
void absoluteDifferenceGPU(float *a, float *b, float *c) {
	c[threadIdx.x] = fabs(a[threadIdx.x] - b[threadIdx.x]);
}

__global__
void absoluteDifferenceGPUBranch(float *a, float *b, float *c, float average) {
	if (a[threadIdx.x] > average) {
		c[threadIdx.x] = fabs(a[threadIdx.x] - b[threadIdx.x]);
	} else {
		c[threadIdx.x] = fabs(b[threadIdx.x] - a[threadIdx.x]);
	}
}

void absoluteDifferenceCPU(float *a, float *b, float *c, int size) {
	for (int i = 0; i < size; ++i) {
		c[i] = fabs(a[i] - b[i]);
	}
}

void absoluteDifferenceCPUBranch(float *a, float *b, float *c, int size, float average) {
	for (int i = 0; i < size; ++i) {
		if (a[i] > average) {
			c[i] = fabs(a[i] - b[i]);
		} else {
			c[i] = fabs(b[i] - a[i]);
		}
	}
}

int main(int argc, char** argv){
	// read command line arguments
	int totalThreads = (1 << 20);
	int blockSize = 256;

	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	float *numsA = new float[totalThreads];
	float *numsB = new float[totalThreads];
	float *results = new float[totalThreads];
	float *branchResults = new float[totalThreads];

	float average = 0;

	// Initialize input vectors
	for (int i = 0; i < totalThreads; ++i) {
		numsA[i] = static_cast<float>(rand()) / RAND_MAX;
		numsB[i] = static_cast<float>(rand()) / RAND_MAX;
		average += numsA[i];
    	average += numsB[i];
	}

	// Average all values
	average /= 2 * totalThreads;

	// Run CPU version and time it
	auto start_cpu = std::chrono::high_resolution_clock::now();
	absoluteDifferenceCPU(numsA, numsB, results, totalThreads);
	auto end_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

	// Run CPU Branch version and time it
	start_cpu = std::chrono::high_resolution_clock::now();
	absoluteDifferenceCPUBranch(numsA, numsB, branchResults, totalThreads, average);
	end_cpu = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> cpu_branch_time = end_cpu - start_cpu;

	// Allocate device memory for GPU version
	float *devNumsA, *devNumsB, *devResults, *devBranchResults;
	cudaMalloc((void**)&devNumsA, totalThreads * sizeof(float));
	cudaMalloc((void**)&devNumsB, totalThreads * sizeof(float));
	cudaMalloc((void**)&devResults, totalThreads * sizeof(float));
	cudaMalloc((void**)&devBranchResults, totalThreads * sizeof(float));

	// Copy data from host to device
	cudaMemcpy(devNumsA, numsA, totalThreads * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devNumsB, numsB, totalThreads * sizeof(float), cudaMemcpyHostToDevice);

	int gridSize = (totalThreads + blockSize - 1) / blockSize;
	float gpu_time_ms = 0;
	float gpu_branch_time_ms = 0;

	cudaEvent_t start_gpu, stop_gpu;
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);

	cudaEventRecord(start_gpu);
	absoluteDifferenceGPU<<<gridSize, blockSize>>>(devNumsA, devNumsB, devResults);
	cudaEventRecord(stop_gpu);

	// Wait for GPU to finish
	cudaEventSynchronize(stop_gpu);
	cudaEventElapsedTime(&gpu_time_ms, start_gpu, stop_gpu);

	// GPU Branching
	cudaEventCreate(&start_gpu);
	cudaEventCreate(&stop_gpu);

	cudaEventRecord(start_gpu);
	absoluteDifferenceGPUBranch<<<gridSize, blockSize>>>(devNumsA, devNumsB, devBranchResults, average);
	cudaEventRecord(stop_gpu);

	// Wait for GPU to finish
	cudaEventSynchronize(stop_gpu);
	cudaEventElapsedTime(&gpu_branch_time_ms, start_gpu, stop_gpu);

	std::cout << "CPU (No branching) time: " << cpu_time.count() << " seconds\n";
	std::cout << "CPU (Branching) time: " << cpu_branch_time.count() << " seconds\n";
	std::cout << "GPU (No branching) time: " << gpu_time_ms / 1000.0 << " seconds\n";
	std::cout << "GPU (Branching) time: " << gpu_branch_time_ms / 1000.0 << " seconds\n";

	cudaFree(devNumsA);
	cudaFree(devNumsB);
	cudaFree(devResults);
	cudaFree(devBranchResults);

    return 0;
}
