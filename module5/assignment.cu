//Based on the work of Andrew Krepps
#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

// Interleaved for the Global Memory
typedef struct {
    float a;
    float b;
    float c;
    float d;
} INTERLEAVED_T;

// Constants for the Constant Memory
typedef unsigned short int u16;
typedef unsigned int u32;

__constant__  static const u32 const_01 = 0x22222222;
__constant__  static const u32 const_02 = 0x55555555;
__constant__  static const u32 const_03 = 0xDDDDDDDD;

// Cuda Time Catcher
__host__ cudaEvent_t get_time(void) {
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time);
    return time;
}

// Absolute Difference for Host Memory
__global__ void absoluteDifference(int n, float *a, float *b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = fabs(a[i] - b[i]);
}

// Host Memeory function
__host__ void host_memory_usage(int blockSize, int totalThreads) {
    float *numsA, *numsB, *devNumsA, *devNumsB;

    cudaMallocHost((void **) &numsA, totalThreads * sizeof(float));
    cudaMallocHost((void **) &numsB, totalThreads * sizeof(float));

    // Initialize input vectors
    for (int i = 0; i < totalThreads; ++i) {
        numsA[i] = static_cast<float>(rand()) / RAND_MAX;
        numsB[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    cudaMalloc((void **) &devNumsA, totalThreads * sizeof(float));
    cudaMalloc((void **) &devNumsB, totalThreads * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(devNumsA, numsA, totalThreads * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(devNumsB, numsB, totalThreads * sizeof(float), cudaMemcpyHostToDevice);

    int gridSize = (totalThreads + blockSize - 1) / blockSize;
    cudaEvent_t start_time = get_time();

    absoluteDifference<<<gridSize, blockSize>>>(totalThreads, devNumsA, devNumsB);

    cudaEvent_t end_time = get_time();
    cudaEventSynchronize(end_time);

    cudaMemcpy(numsB, devNumsB, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);

    float delta = 0;
    cudaEventElapsedTime(&delta, start_time, end_time);

    printf("host_memory (Pinned) elapsed %f seconds\n", delta);

    cudaFree(numsA);
    cudaFree(numsB);
    cudaFree(devNumsA);
    cudaFree(devNumsB);
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
}

// Absolute Difference for Global Memory
__global__ void absoluteDifferenceInterleaved(INTERLEAVED_T * const dest_ptr,
        const INTERLEAVED_T * const src_ptr, const u32 iter,
        const u32 num_elements) {

    const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(tid < num_elements)
    {
        for(u32 i=0; i<iter; i++)
        {
            dest_ptr[tid].a = fabs(src_ptr[tid].a - dest_ptr[tid].a);
            dest_ptr[tid].b = fabs(src_ptr[tid].b - dest_ptr[tid].b);
            dest_ptr[tid].c = fabs(src_ptr[tid].c - dest_ptr[tid].c);
            dest_ptr[tid].d = fabs(src_ptr[tid].d - dest_ptr[tid].d);
        }
    }
}

// Global Memory Function
__host__ void global_memory_usage(int blockSize, int totalThreads) {

    int gridSize = (totalThreads + blockSize - 1) / blockSize;

    float *numsA, *numsB;

    cudaMallocHost((void **) &numsA, totalThreads * sizeof(float));
    cudaMallocHost((void **) &numsB, totalThreads * sizeof(float));

    // Initialize input vectors
    for (int i = 0; i < totalThreads; ++i) {
        numsA[i] = static_cast<float>(rand()) / RAND_MAX;
        numsB[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    INTERLEAVED_T * devNumsB;
    INTERLEAVED_T * devNumsA;

    cudaMalloc((void **) &devNumsA, totalThreads * sizeof(float));
    cudaMalloc((void **) &devNumsB, totalThreads * sizeof(float));

    cudaMemcpy(devNumsA, numsA, totalThreads * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(devNumsB, numsB, totalThreads * sizeof(float),cudaMemcpyHostToDevice);

    cudaEvent_t start_time = get_time();

    absoluteDifferenceInterleaved<<<gridSize, blockSize>>>(devNumsB, devNumsA, 1, sizeof(numsA));

    cudaEvent_t end_time = get_time();
    cudaEventSynchronize(end_time);

    cudaMemcpy(numsB, devNumsB, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);

    float delta = 0.0F;
    cudaEventElapsedTime(&delta, start_time, end_time);
    printf("global_memory elapsed %f seconds\n", delta);

    cudaFree(numsA);
    cudaFree(numsB);
    cudaFree(devNumsA);
    cudaFree(devNumsB);
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
}

__device__ void copy_data_to_shared(const float * const data,
                                    float * const sort_tmp,
                                    const int num_elements,
                                    const int tid)
{
    // Copy data into temp store
    for(int i = 0; i<num_elements; i++)
    {
        sort_tmp[i+tid] = data[i+tid];
    }
    __syncthreads();
}

__global__ void sharedAbsoluteDifference(float * data, float *result, int totalThreads) {
    // Shared memory for this block
    __shared__ float * sharedData;
    cudaMalloc((void** ) &sharedData, sizeof(float) * totalThreads);

    const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Load data from global memory into shared memory
    copy_data_to_shared(data, sharedData, sizeof(float) * totalThreads, tid);

    // Perform reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] = fabs(sharedData[tid + stride] - sharedData[tid]);
        }
        __syncthreads();  // Ensure all threads finish this step before continuing
    }

    // Write the result of this block back to global memory
    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

// Shared Memory Function
__host__ void shared_memory_usage(int blockSize, int totalThreads) {
    float *data = NULL;
    float *result = NULL;

    const u32 num_threads = totalThreads;
    const u32 num_blocks = (totalThreads + blockSize - 1) / blockSize;

    cudaMalloc((void** ) &data, sizeof(float) * totalThreads);
    cudaMalloc((void** ) &result, sizeof(float) * totalThreads);

    cudaEvent_t start_time = get_time();

    sharedAbsoluteDifference<<<num_blocks,num_threads>>>(data, result, totalThreads);


    cudaEvent_t end_time = get_time();

    cudaDeviceSynchronize();
    cudaEventSynchronize(end_time);

    float delta = 0.0F;
    cudaEventElapsedTime(&delta, start_time, end_time);
    printf("shared_memory elapsed %f seconds\n", delta);
    cudaFree((void* ) data);
}

// GPU Constant Memory Usage
__global__ void const_binary_calcs(u32 * const data, const u32 num_elements) {
    const u32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < num_elements) {
        u32 d = const_01;

        for (int i = 0; i < num_elements; i++) {
            d ^= const_02;
            d |= const_01;
            d &= const_03;
        }

        data[tid] = d;
    }
}

// Constant Memory Function
__host__ void constant_memory_usage(int blockSize, int totalThreads) {
    u32 *data = NULL;
    const u32 num_threads = totalThreads;
    const u32 num_blocks = (totalThreads + blockSize - 1) / blockSize;

    cudaMalloc((void** ) &data, sizeof(int) * totalThreads);

    cudaEvent_t start_time = get_time();

    const_binary_calcs<<<num_blocks,num_threads>>>(data, totalThreads);
    cudaEvent_t end_time = get_time();

    cudaDeviceSynchronize();
    cudaEventSynchronize(end_time);

    float delta = 0.0F;
    cudaEventElapsedTime(&delta, start_time, end_time);
    printf("constant_memory elapsed %f seconds\n", delta);
    cudaFree((void* ) data);
}

__global__ void registerSqauredProduct(unsigned int * const data, const unsigned int num_elements)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(tid < num_elements)
    {
        unsigned int d_tmp = data[tid];
        d_tmp = d_tmp * 8;
        d_tmp = d_tmp * d_tmp;
        data[tid] = d_tmp;
    }
}

// Register Memory Function
__host__ void register_memory_usage(int blockSize, int totalThreads) {
    const u32 num_threads = totalThreads;
    const u32 num_blocks = (totalThreads + blockSize - 1) / blockSize;
    u32 * host_packed_array;
    u32 * data_gpu;

    cudaMallocHost((void **) &host_packed_array, totalThreads * sizeof(u32));
    cudaMalloc(&data_gpu, totalThreads * sizeof(u32));

    // Initialize input vectors
    for (int i = 0; i < totalThreads; ++i) {
        host_packed_array[i] = static_cast<u32>(rand() % (100 + 1 - 1) + 1);

    }

    cudaMemcpy(data_gpu, host_packed_array, totalThreads * sizeof(u32),cudaMemcpyHostToDevice);
    cudaEvent_t start_time = get_time();

    registerSqauredProduct<<<num_blocks, num_threads>>>(data_gpu, totalThreads);
    cudaEvent_t end_time = get_time();

    cudaDeviceSynchronize();
    cudaEventSynchronize(end_time);

    float delta = 0.0F;
    cudaEventElapsedTime(&delta, start_time, end_time);
    printf("register_memory elapsed %f seconds\n", delta);

    cudaFree((void* ) data_gpu);
}

// Main
int main(int argc, char **argv) {
    // read command line arguments
    int totalThreads = (1 << 20);
    int blockSize = 256;

    if (argc >= 2) {
        totalThreads = atoi(argv[1]);
    }
    if (argc >= 3) {
        blockSize = atoi(argv[2]);
    }

    int numBlocks = totalThreads / blockSize;

    // validate command line arguments
    if (totalThreads % blockSize != 0) {
        ++numBlocks;
        totalThreads = numBlocks * blockSize;

        printf("Warning: Total thread count is not evenly divisible by the block size\n");
        printf("The total number of threads will be rounded up to %d\n", totalThreads);
    }

    host_memory_usage(blockSize, totalThreads);
    global_memory_usage(blockSize, totalThreads);
    shared_memory_usage(blockSize, totalThreads);
    constant_memory_usage(blockSize, totalThreads);
    register_memory_usage(blockSize, totalThreads);

    cudaDeviceReset();

    return 0;
}
