
#include "src/Utils.h"
#include <chrono> // For timing
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define MAX_NUM 10
#define MIN_NUM -10
void CpuMatMul(float* MatrixA, float* MatrixB, float* OutMatrix, int MatrixSize)
{
	for(int i = 0; i < MatrixSize; i++)
	{
		for(int j = 0; j < MatrixSize; j++)
		{
			float sum = 0.0;
			for(int k = 0; k < MatrixSize; k++)
			{
				sum += MatrixA[i * MatrixSize + k] * MatrixB[k * MatrixSize + j];
			}
			OutMatrix[i * MatrixSize + j] = sum;
		}
	}
}

__global__ void GpuMatMul(float* MatrixA, float* MatrixB, float* OutMatrix, int MatrixSize)

{
	// Working on C[i,j]

	//INFO: Bellow is some info on the cuda params
	/*
    blockDim is the number of threads in this block
    blockIDx is the coordinate OF THE BLOCK that THIS THREAD is IN
    threadIDx is the coordinate OF THE THREAD inside this block
    */

	int i = blockDim.y * blockIdx.y + threadIdx.y; //essentially the *row*
	int j = blockDim.x * blockIdx.x + threadIdx.x; //essentially the *column*

	// Parallel mat mul
	if(i < MatrixSize && j < MatrixSize)
	{
		// Value at C[i,j]
		float value = 0;
		for(int k = 0; k < MatrixSize; k++)
		{
			value += MatrixA[i * MatrixSize + k] * MatrixB[k * MatrixSize + j];
		}

		// Assigning calculated value
		OutMatrix[i * MatrixSize + j] = value;
	}
}

__global__ void GpuTranspose(float* MatrixA, float* OutMatrix, int MatrixSize)

{
	// Working on C[i,j]

	//INFO: Bellow is some info on the cuda params
	/*
    blockDim is the number of threads in this block
    blockIDx is the coordinate OF THE BLOCK that THIS THREAD is IN
    threadIDx is the coordinate OF THE THREAD inside this block
    */

	int i = blockDim.y * blockIdx.y + threadIdx.y; //essentially the *row*
	int j = blockDim.x * blockIdx.x + threadIdx.x; //essentially the *column*

	if(i < MatrixSize && j < MatrixSize)
	{
		int currK = i * MatrixSize + j;
		int newK = j * MatrixSize + i;

		OutMatrix[newK] = MatrixA[currK];
	}
}
__global__ void transposedMatrixKernel(float* d_a, float* d_b, int N)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int j = threadIdx.y + blockDim.y * blockIdx.y;

	while(i < N)
	{
		j = threadIdx.y + blockDim.y * blockIdx.y;
		while(j < N)
		{
			d_b[i * N + j] = d_a[j * N + i];
			j += blockDim.y * gridDim.y;
		}
		i += blockDim.x * gridDim.x;
	}
}

int main()
{
	// MyCudaUtils::OutputGpuInformation();

	int matrixSize = 5;

	//INFO: CPU/Host matrix initialisation
	float* matrixA = (float*)malloc(matrixSize * matrixSize * sizeof(float));
	float* matrixB = (float*)malloc(matrixSize * matrixSize * sizeof(float));
	MyCudaUtils::RandomFillMatricies(matrixA, matrixB, matrixSize, MAX_NUM, MIN_NUM);
	float* matrixC = (float*)malloc(matrixSize * matrixSize * sizeof(float));

	//INFO: Benchmark CPU performance
	auto start = std::chrono::high_resolution_clock::now();
	// CpuMatMul(matrixA, matrixB, matrixC, matrixSize);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> cpuElapsed = end - start;
	printf("CPU MatMul time taken: %.3f ms\n", cpuElapsed.count());

	// MyCudaUtils::PrintMatrix("Matrix A", matrixA, matrixSize);
	// MyCudaUtils::PrintMatrix("Matrix B", matrixB, matrixSize);
	// MyCudaUtils::PrintMatrix("Matrix C (Result)", matrixC, matrixSize);

	//INFO: GPU/Device initialisation
	float* d_A;
	float* d_B;
	float* d_C;
	cudaError_t err_A = cudaMalloc((void**)&d_A, matrixSize * matrixSize * sizeof(float));
	cudaError_t err_B = cudaMalloc((void**)&d_B, matrixSize * matrixSize * sizeof(float));
	cudaError_t err_C = cudaMalloc((void**)&d_C, matrixSize * matrixSize * sizeof(float));

	// Copying A and B to device memory
	cudaError_t err_A_ =
		cudaMemcpy(d_A, matrixA, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaError_t err_B_ =
		cudaMemcpy(d_B, matrixB, matrixSize * matrixSize * sizeof(float), cudaMemcpyHostToDevice);

	//INFO: Kernel setup
	// dim_block is the number of threads IN each BLOCK
	dim3 dim_block(32, 32, 1);
	// dim_grid is the number of blocks IN THE GRID.
	dim3 dim_grid(ceil(matrixSize / 32.0), ceil(matrixSize / 32.0), 1);

	//INFO: Benchmark GPU performance
	start = std::chrono::high_resolution_clock::now();
	GpuMatMul<<<dim_grid, dim_block>>>(d_A, d_B, d_C, matrixSize);
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> gpuElapsed = end - start;
	printf("GPU MatMul time taken: %.3f ms\n", gpuElapsed.count());

	// Copy back results
	cudaError_t err_C_ =
		cudaMemcpy(matrixC, d_C, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
	MyCudaUtils::PrintMatrix("Matrix before transpose", matrixC, matrixSize);

	//INFO: Output perf metrics
	double speedImprovement = cpuElapsed / gpuElapsed;
	printf("Speed Improvement (CPU/GPU): %.2f x\n", speedImprovement);

	//INFO: Transpose Benchmark GPU performance
	// start = std::chrono::high_resolution_clock::now();
	// GpuTranspose<<<dim_grid, dim_block>>>(d_C, d_C, matrixSize);
	// end = std::chrono::high_resolution_clock::now();
	// std::chrono::duration<double, std::milli> gpuTransposeElapsed = end - start;
	// printf("naive GPU MatMul time taken: %.3f ms\n", gpuTransposeElapsed.count());

	// //INFO: Transpose Benchmark GPU performance *sourced code*
	// start = std::chrono::high_resolution_clock::now();
	// transposedMatrixKernel<<<dim_grid, dim_block>>>(d_C, d_C, matrixSize);
	// end = std::chrono::high_resolution_clock::now();
	// std::chrono::duration<double, std::milli> gpuTranspose2Elapsed = end - start;
	// printf("sourced GPU MatMul time taken: %.3f ms\n", gpuTranspose2Elapsed.count());
	//

	err_C_ =
		cudaMemcpy(matrixC, d_C, matrixSize * matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
	MyCudaUtils::PrintMatrix("Matrix after transpose", matrixC, matrixSize);
	//INFO: Free all allocated memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(matrixA);
	free(matrixB);
	free(matrixC);

	return 0;
}
