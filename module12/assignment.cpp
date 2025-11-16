#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>

#define CHECK_ERR(err, msg) \
    if (err != CL_SUCCESS) { \
        std::cerr << msg << " failed with error " << err << std::endl; \
        std::exit(1); \
    }

std::string load_file(const std::string& path) {
    std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << path << std::endl;
        std::exit(1);
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

int main(int argc, char** argv) {
    int N = 16;  // default: 16 elements

if (argc >= 2) {
        N = atoi(argv[1]);
    }

    if (N <= 0) {
        std::cerr << "N must be positive" << std::endl;
        return 1;
    }

    std::cout << "Running polynomial_transform on N = " << N << " elements\n";

    cl_int err;

    // Get platform
    cl_uint numPlatforms = 0;
    CHECK_ERR(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs");
    if (numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found\n";
        return 1;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    CHECK_ERR(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr),
              "clGetPlatformIDs(2)");

    cl_platform_id platform = platforms[0]; // pick first

    // Get device
    cl_device_id device = nullptr;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "No GPU device found, trying CPU...\n";
        CHECK_ERR(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr),
                  "clGetDeviceIDs CPU");
    }

    // Create context and command queue
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_ERR(err, "clCreateContext");

    cl_command_queue queue =
        clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    CHECK_ERR(err, "clCreateCommandQueue");

    // Load and build program
    std::string src = load_file("polynomial_transform.cl");
    const char* srcPtr = src.c_str();
    size_t srcLen = src.size();

    cl_program program = clCreateProgramWithSource(context, 1, &srcPtr, &srcLen, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Print build log
        size_t logSize = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build failed:\n" << log.data() << std::endl;
        CHECK_ERR(err, "clBuildProgram");
    }

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "poly_transform", &err);
    CHECK_ERR(err, "clCreateKernel");

    // Host data initialization
    std::vector<float> h_input(N);
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);  // x = i
    }

    std::vector<cl_float4> h_output(N);

    // Device buffers (full range)
    cl_mem d_in = clCreateBuffer(context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * N,
                                 h_input.data(),
                                 &err);
    CHECK_ERR(err, "clCreateBuffer d_in");

    cl_mem d_out = clCreateBuffer(context,
                                  CL_MEM_WRITE_ONLY,
                                  sizeof(cl_float4) * N,
                                  nullptr,
                                  &err);
    CHECK_ERR(err, "clCreateBuffer d_out");

    // Sub-buffer setup: split into two halves
    int N_first  = N / 2;
    int N_second = N - N_first;

    // region for second half of input
    cl_buffer_region in_region;
    in_region.origin = sizeof(float) * N_first;
    in_region.size   = sizeof(float) * N_second;

    cl_mem d_in_sub = clCreateSubBuffer(d_in,
                                        CL_MEM_READ_ONLY,
                                        CL_BUFFER_CREATE_TYPE_REGION,
                                        &in_region,
                                        &err);
    CHECK_ERR(err, "clCreateSubBuffer d_in_sub");

    // region for second half of output
    cl_buffer_region out_region;
    out_region.origin = sizeof(cl_float4) * N_first;
    out_region.size   = sizeof(cl_float4) * N_second;

    cl_mem d_out_sub = clCreateSubBuffer(d_out,
                                         CL_MEM_WRITE_ONLY,
                                         CL_BUFFER_CREATE_TYPE_REGION,
                                         &out_region,
                                         &err);
    CHECK_ERR(err, "clCreateSubBuffer d_out_sub");

    //First kernel launch: process FIRST HALF using main buffers
    {
        int n_first = N_first;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
        err |= clSetKernelArg(kernel, 2, sizeof(int),    &n_first);
        CHECK_ERR(err, "clSetKernelArg first half");

        size_t globalSize[1] = { static_cast<size_t>(n_first) };
        err = clEnqueueNDRangeKernel(queue,
                                     kernel,
                                     1,
                                     nullptr,
                                     globalSize,
                                     nullptr,
                                     0,
                                     nullptr,
                                     nullptr);
        CHECK_ERR(err, "clEnqueueNDRangeKernel first half");
    }

    // Second kernel launch: process SECOND HALF via sub-buffers
    {
        int n_second = N_second;
        err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in_sub);
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out_sub);
        err |= clSetKernelArg(kernel, 2, sizeof(int),    &n_second);
        CHECK_ERR(err, "clSetKernelArg second half");

        size_t globalSize[1] = { static_cast<size_t>(n_second) };
        err = clEnqueueNDRangeKernel(queue,
                                     kernel,
                                     1,
                                     nullptr,
                                     globalSize,
                                     nullptr,
                                     0,
                                     nullptr,
                                     nullptr);
        CHECK_ERR(err, "clEnqueueNDRangeKernel second half");
    }

    // Read back results
    err = clEnqueueReadBuffer(queue,
                              d_out,
                              CL_TRUE,  // blocking
                              0,
                              sizeof(cl_float4) * N,
                              h_output.data(),
                              0,
                              nullptr,
                              nullptr);
    CHECK_ERR(err, "clEnqueueReadBuffer");

    // Print a the results to prove it worked
    int toPrint = N;
    std::cout << "First " << toPrint << " results (x, x^2, x^3, sin(x)):\n";
    for (int i = 0; i < toPrint; ++i) {
        float x  = h_input[i];
        float x1 = h_output[i].s[0];
        float x2 = h_output[i].s[1];
        float x3 = h_output[i].s[2];
        float s  = h_output[i].s[3];

        std::cout << "x=" << x
                  << " -> (" << x1 << "[x], " << x2 << "[x^2], " << x3 << "[x^3], " << s << "[sin(x)])\n";
    }

    // Cleanup
    clReleaseMemObject(d_in_sub);
    clReleaseMemObject(d_out_sub);
    clReleaseMemObject(d_in);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
