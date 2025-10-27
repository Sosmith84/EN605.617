#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <sstream>
#include <iomanip>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define CUDA_CHECK(call) do { \
	cudaError_t err = (call); \
	if (err != cudaSuccess) { \
    	std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) \
              << " at line " << __LINE__ << std::endl; std::exit(1);} \
} while(0)

#define CUSOLVER_CHECK(call) do { \
	cusolverStatus_t st = (call); \
	if (st != CUSOLVER_STATUS_SUCCESS) { \
    	std::cerr << "[CUSOLVER ERROR] at line " << __LINE__ << std::endl; std::exit(1);} \
} while(0)

#define CUSPARSE_CHECK(call) do { \
	cusparseStatus_t st = (call); \
	if (st != CUSPARSE_STATUS_SUCCESS) { \
    	std::cerr << "[CUSPARSE ERROR] at line " << __LINE__ << std::endl; std::exit(1);} \
} while(0)

// Small device functors
// Get the absolute Value
struct AbsVal {
	__host__ __device__ float operator()(float x) const { return fabsf(x); }
};

// Get the Charge Density
struct ComputeChargeDensity {
	int w, h;
	float h2, eps;
	ComputeChargeDensity(int width, int height, float spacing, float epsilon)
  			: w(width), h(height), h2(spacing*spacing), eps(epsilon) {}
	__host__ __device__ float operator()(int idx) const {
	    int x = idx % w, y = idx / w;
	    if (x == w/2 && y == h/2) return h2/eps; // 1 point charge at center
	    return 0.0f;
  }
};

// Assembly: 5-point Laplacian
void assembleLaplacianCSR(int width, int height,
                          std::vector<int>& rowOffsets,
                          std::vector<int>& colIndices,
                          std::vector<float>& values) {
	const int N = width * height;
	rowOffsets.assign(N+1, 0);
	colIndices.clear(); values.clear(); colIndices.reserve(5*N); values.reserve(5*N);

	for (int j = 0; j < height; ++j) {
	    for (int i = 0; i < width; ++i) {
		    int r = j*width + i;
		    if (i > 0) {
				colIndices.push_back(r-1);      values.push_back(1.0f);
	 		}
		    if (j > 0) {
				colIndices.push_back(r-width);
				values.push_back(1.0f);
			}

			colIndices.push_back(r);
			values.push_back(-4.0f);

		    if (i < width-1) {
				colIndices.push_back(r+1);
				values.push_back(1.0f); }
		    if (j < height-1) {
				colIndices.push_back(r+width);
  				values.push_back(1.0f); }
		    rowOffsets[r+1] = static_cast<int>(colIndices.size());
	    }
  }
}

// Device allocations & copies
void allocateAndCopyToDevice(const std::vector<int>& rowOffsets,
                             const std::vector<int>& colIndices,
                             const std::vector<float>& values,
                             int totalNodes, int nnz,
                             int*& dRow, int*& dCol, float*& dVal,
                             float*& dRhs, float*& dPhi) {
	CUDA_CHECK(cudaMalloc(&dRow, (totalNodes+1)*sizeof(int)));
	CUDA_CHECK(cudaMalloc(&dCol, nnz*sizeof(int)));
	CUDA_CHECK(cudaMalloc(&dVal, nnz*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dRhs, totalNodes*sizeof(float)));
	CUDA_CHECK(cudaMalloc(&dPhi, totalNodes*sizeof(float)));

	CUDA_CHECK(cudaMemcpy(dRow, rowOffsets.data(), (totalNodes+1)*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dCol, colIndices.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(dVal, values.data(), nnz*sizeof(float), cudaMemcpyHostToDevice));
}

// Build RHS on device
void buildRHSOnDevice(int width, int height,
                      float gridSpacing, float epsilon0,
                      float* dRhs) {
    const int N = width*height;
    thrust::device_ptr<float> rhsPtr(dRhs);
    thrust::transform(thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(N),
                    rhsPtr,
                    ComputeChargeDensity(width, height, gridSpacing, epsilon0));
}

// Solve A*phi = b using cuSolver sparse Cholesky
void solveWithCholesky(int totalNodes, int nnz,
                       int* dRow, int* dCol, float* dVal,
                       float* dRhs, float* dPhi, float tol) {
    cusolverSpHandle_t solver;   CUSOLVER_CHECK(cusolverSpCreate(&solver));
    cusparseMatDescr_t descrA;   CUSPARSE_CHECK(cusparseCreateMatDescr(&descrA));
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int singularity = 0;
    const int reorder = 1;
    std::cout << "[INFO] cuSolverSpScsrlsvchol...\n";
    CUSOLVER_CHECK(cusolverSpScsrlsvchol(solver, totalNodes, nnz,
                   descrA, dVal, dRow, dCol, dRhs, tol, reorder, dPhi, &singularity));
    if (singularity == -1) {
		std::cout << "[INFO] Success\n";
	} else {
        std::cout << "[WARN] Singular at row " << singularity << "\n";
	}

    cusolverSpDestroy(solver);
    cusparseDestroyMatDescr(descrA);
}

// Compute average |residual| = mean(|A*phi - b|)
float computeResidualAverage(int totalNodes, int nnz,
                             int* dRow, int* dCol, float* dVal,
                             float* dPhi, float* dRhs) {
    cusparseHandle_t h; CUSPARSE_CHECK(cusparseCreate(&h));
    // Create descriptors for SpMV
    cusparseSpMatDescr_t A;
    cusparseDnVecDescr_t x, y;
    CUSPARSE_CHECK(cusparseCreateCsr(&A, totalNodes, totalNodes, nnz,
                      dRow, dCol, dVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    thrust::device_vector<float> phi(totalNodes), ytmp(totalNodes), rhs(totalNodes);
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(phi.data()), dPhi, totalNodes*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(rhs.data()), dRhs, totalNodes*sizeof(float), cudaMemcpyDeviceToDevice));

    CUSPARSE_CHECK(cusparseCreateDnVec(&x, totalNodes, thrust::raw_pointer_cast(phi.data()), CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&y, totalNodes, thrust::raw_pointer_cast(ytmp.data()), CUDA_R_32F));

    float alpha = 1.0f, beta = 0.0f;
    size_t bufSz=0; void* dBuf=nullptr;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      &alpha, A, x, &beta, y, CUDA_R_32F,
                      CUSPARSE_SPMV_ALG_DEFAULT, &bufSz));
    CUDA_CHECK(cudaMalloc(&dBuf, bufSz));
    CUSPARSE_CHECK(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      &alpha, A, x, &beta, y, CUDA_R_32F,
                      CUSPARSE_SPMV_ALG_DEFAULT, dBuf));
    CUDA_CHECK(cudaFree(dBuf));

    // residual = ytmp - rhs
    thrust::transform(ytmp.begin(), ytmp.end(), rhs.begin(), ytmp.begin(), thrust::minus<float>());
    float avg = thrust::transform_reduce(ytmp.begin(), ytmp.end(), AbsVal(), 0.0f, thrust::plus<float>()) / totalNodes;

    cusparseDestroyDnVec(x); cusparseDestroyDnVec(y);
    cusparseDestroySpMat(A); cusparseDestroy(h);
    return avg;
}

// Utility: fetch potential at center
float fetchCenterPotential(int width, int height, float* dPhi) {
    int cx = width/2, cy = height/2;
    float val=0.f;
    CUDA_CHECK(cudaMemcpy(&val, dPhi + cy*width + cx, sizeof(float), cudaMemcpyDeviceToHost));
    return val;
}

void writeHeatmapPNG(const std::string& path, const std::vector<float>& phi, int W, int H) {
	std::vector<unsigned char> img(W * H * 3);

	float minv = *std::min_element(phi.begin(), phi.end());
	float maxv = *std::max_element(phi.begin(), phi.end());
	float scale = (maxv - minv > 1e-9f) ? 1.0f / (maxv - minv) : 1.0f;

	for (int j = 0; j < H; ++j) {
		for (int i = 0; i < W; ++i) {
			float v = (phi[j*W + i] - minv) * scale; // normalize [0,1]
			int idx = (j*W + i) * 3;
			// simple blue→red gradient
			img[idx+0] = (unsigned char)(v * 255);         // R
			img[idx+1] = (unsigned char)(0);               // G
			img[idx+2] = (unsigned char)((1.0f - v)*255);  // B
		}
	}

	if (stbi_write_png(path.c_str(), W, H, 3, img.data(), W * 3)) {
		std::cout << "[INFO] Wrote heatmap to " << path << "\n";
	} else {
		std::cerr << "[ERROR] Failed to write " << path << "\n";
	}
}

//  Cleanup
void freeDeviceMemory(int* dRow, int* dCol, float* dVal,
                      float* dRhs, float* dPhi) {
    cudaFree(dRow); cudaFree(dCol); cudaFree(dVal);
    cudaFree(dRhs); cudaFree(dPhi);
}

static void printHelp() {
	std::cout <<
	"Usage: ./electric_potential [options]\n"
	"Options:\n"
	"  --grid <N>     Set the grid to NxN points (default: 64x64)\n"
	"  --real         Use the real vacuum permittivity constant \u03B50 = 8.854e-12 F/m\n"
	"  --tol <T>      Solver tolerance (default 1e-6)\n"
	"  --help         Show this help message\n\n"
	"Examples:\n"
	"  ./electric_potential --grid 128\n"
	"      - run a 128x128 simulation with normalized \u03B50 = 1.0\n\n"
	"  ./electric_potential --real\n"
	"      - use the physical constant \u03B50 = 8.854e-12 F/m\n";
}

int main(int argc, char* argv[]) {
    int gridWidth = 64;
    int gridHeight = 64;
    float epsilon0 = 1.0f; // normalized
	float tol = 1e-6f;

    // Parse command-line args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--grid" && i + 1 < argc) {
            gridWidth = gridHeight = std::stoi(argv[++i]);
        } else if (arg == "--real") {
            epsilon0 = 8.854e-12f;
   		} else if (arg == "--tol" && i + 1 < argc) {
    		tol = std::stof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            printHelp();
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printHelp();
            return 1;
        }
    }

    const int totalNodes = gridWidth * gridHeight;
    const float gridSpacing = 1.0f / gridWidth;

    std::cout << "[INIT] Building " << gridWidth << "x" << gridHeight
            << " Laplacian (" << totalNodes << " nodes)\n";

    // Host assembly
    std::vector<int> rowOffsets; std::vector<int> colIndices; std::vector<float> values;
    assembleLaplacianCSR(gridWidth, gridHeight, rowOffsets, colIndices, values);
    int nnz = static_cast<int>(values.size());
    std::cout << "[DEBUG] nnz = " << nnz << "\n";

    // Device alloc & copy
    int *dRow=nullptr, *dCol=nullptr; float *dVal=nullptr, *dRhs=nullptr, *dPhi=nullptr;
    allocateAndCopyToDevice(rowOffsets, colIndices, values,
                    totalNodes, nnz, dRow, dCol, dVal, dRhs, dPhi);

    // Build RHS (charge)
    buildRHSOnDevice(gridWidth, gridHeight, gridSpacing, epsilon0, dRhs);

    // Solve
    solveWithCholesky(totalNodes, nnz, dRow, dCol, dVal, dRhs, dPhi, tol);

    // Diagnostics
    float avgRes = computeResidualAverage(totalNodes, nnz, dRow, dCol, dVal, dPhi, dRhs);
    float center = fetchCenterPotential(gridWidth, gridHeight, dPhi);
    std::cout << "[INFO] Average |residual| = " << avgRes << "\n";
    std::cout << "[INFO] Center potential = " << center << "\n";
    std::cout << "[INFO] Simulation complete.\n";

	// Copy potential back to host
	std::vector<float> phiHost(totalNodes);
	CUDA_CHECK(cudaMemcpy(phiHost.data(), dPhi, totalNodes*sizeof(float), cudaMemcpyDeviceToHost));

	std::string heatmap = "potential_" + std::to_string(gridHeight) + ".png";
	// Create heatmap
	writeHeatmapPNG(heatmap, phiHost, gridWidth, gridHeight);

    // Cleanup
    freeDeviceMemory(dRow, dCol, dVal, dRhs, dPhi);
    return 0;
}
