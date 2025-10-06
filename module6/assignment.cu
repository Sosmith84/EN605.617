//Based on the work of Andrew Krepps
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>

// Cuda Time Catcher
__host__ cudaEvent_t get_time(void) {
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time);
    return time;
}

// Performs some mathematical operations
__global__ void gpu_computation(const float* in, float* out, size_t N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        float x = in[tid];
        float a = __sinf(x);
        float b = __expf(0.5f * x);
        float c = fmaf(a, b, 1.234567f);
        float d = powf(c + 0.1f, 1.37f);
        float result = logf(fabsf(x) + 1.0f);
        out[tid] = result;
    }
}

__host__ void stream_event_demo(int blockSize, int totalThreads) {
    float* host_in;
    float* host_out;

    // You could take this as an input from command line to increase/decrease the number of Streams
    int numStreams = 4;

    cudaMallocHost((void **) &host_in, totalThreads * sizeof(float));
    cudaMallocHost((void **) &host_out, totalThreads * sizeof(float));

    // Create streams
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Initialize input vectors
    for (int i = 0; i < totalThreads; ++i) {
        host_in[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Create events for timing and synchronization
    std::vector<cudaEvent_t> startHostToDevice(numStreams), endHostToDevice(numStreams), startKernel(numStreams), endKernel(numStreams);
    std::vector<cudaEvent_t> startDeviceToHost(numStreams), endDeviceToHost(numStreams), streamDone(numStreams);

    // Create the events for each of the streams
    for (int i = 0; i < numStreams; ++i) {
        cudaEventCreate(&startHostToDevice[i]);
        cudaEventCreate(&endHostToDevice[i]);
        cudaEventCreate(&startKernel[i]);
        cudaEventCreate(&endKernel[i]);
        cudaEventCreate(&startDeviceToHost[i]);
        cudaEventCreate(&endDeviceToHost[i]);
        cudaEventCreate(&streamDone[i]);
    }

    std::vector<float *> device_in(numStreams, nullptr), device_out(numStreams, nullptr);

    // Divide the total number of elements between the four Streams
    size_t chunkSize = (totalThreads + numStreams - 1) / numStreams;
    
    printf("Total Elements: %d, Block Size: %d, Estimated Chunk size per Stream: %zd\n\n", totalThreads, blockSize, chunkSize);

    printf("Per Stream Chunk Size:\n");
    std::vector<size_t> chunkOffsets(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        chunkOffsets[i] = i * chunkSize;
        size_t thisTotal = std::min(chunkSize, (i == numStreams-1) ? (totalThreads - chunkOffsets[i]) : chunkSize);
        if (thisTotal == 0) break;
        cudaMalloc(&device_in[i], thisTotal * sizeof(float));
        cudaMalloc(&device_out[i], thisTotal * sizeof(float));
    }

    // Timing
    cudaEvent_t gpuStart, gpuStop;

    // Start GPU and CPU timers
    gpuStart = get_time();
    auto cpuStart = std::chrono::high_resolution_clock::now();

    // Launch per stream
    for (int i = 0; i < numStreams; ++i) {
        size_t offset = chunkOffsets[i];
        size_t thisN = std::min(chunkSize, totalThreads - offset);
        if (thisN == 0) {
            cudaEventRecord(streamDone[i], streams[i]);
            continue;
        }

        // HostToDevice
        cudaEventRecord(startHostToDevice[i], streams[i]);
        cudaMemcpyAsync(device_in[i], host_in + offset, thisN * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaEventRecord(endHostToDevice[i], streams[i]);

        // Kernel
        int blocks = (int)((thisN + blockSize - 1) / blockSize);
        cudaEventRecord(startKernel[i], streams[i]);

        gpu_computation<<<blocks, blockSize, 0, streams[i]>>>(device_in[i], device_out[i], thisN);

        cudaEventRecord(endKernel[i], streams[i]);

        // DeviceToHost
        cudaEventRecord(startDeviceToHost[i], streams[i]);
        cudaMemcpyAsync(host_out + offset, device_out[i], thisN * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        cudaEventRecord(endDeviceToHost[i], streams[i]);

        cudaEventRecord(streamDone[i], streams[i]);
    }

    // Syncronize all the Streams
    for (int i = 0; i < numStreams; ++i) {
        cudaEventSynchronize(streamDone[i]);
    }

    // Set the Stop timers
    gpuStop = get_time();
    auto cpuStop = std::chrono::high_resolution_clock::now();

    // Calculate the Per-Stream Times and Total Times
    float totalHostToDevice = 0.0f, totalKernel = 0.0f, totalDeviceToHost = 0.0f;
    for (int i = 0; i < numStreams; ++i) {
        float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
        cudaEventElapsedTime(&h2d_ms, startHostToDevice[i], endHostToDevice[i]);
        cudaEventElapsedTime(&kernel_ms, startKernel[i], endKernel[i]);
        cudaEventElapsedTime(&d2h_ms, startDeviceToHost[i], endDeviceToHost[i]);
        totalHostToDevice += h2d_ms;
        totalKernel += kernel_ms;
        totalDeviceToHost += d2h_ms;
        printf(" Stream %2d: HostToDevice = %7.3f  Kernel = %7.3f  DeviceToHost = %7.3f   (chunkSize=%zu)\n",
               i, h2d_ms, kernel_ms, d2h_ms,
               std::min(chunkSize, (i == numStreams-1) ? (totalThreads - chunkOffsets[i]) : chunkSize));
    }

    float totalGPUms = 0.0f;
    cudaEventElapsedTime(&totalGPUms, gpuStart, gpuStop);
    double cpuMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpuStop - cpuStart).count();

    printf("\nTotal Sums of stream timings: HostToDevice=%.3f ms  Kernel=%.3f ms  DeviceToHost=%.3f ms\n",
           totalHostToDevice, totalKernel, totalDeviceToHost);
    printf("Total GPU time: %.3f ms\n", totalGPUms);
    printf("Total CPU time: %.3f ms\n", cpuMs);

    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        if (device_in[i]) cudaFree(device_in[i]);
        if (device_out[i]) cudaFree(device_out[i]);
    }
    for (int i = 0; i < numStreams; ++i) {
        cudaEventDestroy(startHostToDevice[i]);
        cudaEventDestroy(endHostToDevice[i]);
        cudaEventDestroy(startKernel[i]);
        cudaEventDestroy(endKernel[i]);
        cudaEventDestroy(startDeviceToHost[i]);
        cudaEventDestroy(endDeviceToHost[i]);
        cudaEventDestroy(streamDone[i]);
    }
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);
    for (int i = 0; i < numStreams; ++i) cudaStreamDestroy(streams[i]);

    cudaFreeHost(host_in);
    cudaFreeHost(host_out);
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

    // Start the demo
    stream_event_demo(blockSize, totalThreads);

    return 0;
}
