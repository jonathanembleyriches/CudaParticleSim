#include "Utils.h"

#include <iostream>
void MyCudaUtils::OutputGpuInformation()
{

	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);

	if(err != cudaSuccess)
	{
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
		return;
	}

	if(deviceCount == 0)
	{
		std::cout << "No CUDA-capable GPU detected." << std::endl;
		return;
	}

	for(int i = 0; i < deviceCount; ++i)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);

		std::cout << "GPU #" << i << " Information:" << std::endl;
		std::cout << "  Name: " << deviceProp.name << std::endl;
		std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor
				  << std::endl;
		std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB"
				  << std::endl;
		std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
		std::cout << "  Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
		std::cout << "  Memory Clock Rate: " << deviceProp.memoryClockRate / 1000 << " MHz"
				  << std::endl;
		std::cout << "  Memory Bus Width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
		std::cout << "  L2 Cache Size: " << deviceProp.l2CacheSize / 1024 << " KB" << std::endl;
		std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
		std::cout << "  Max Grid Size: [" << deviceProp.maxGridSize[0] << ", "
				  << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << "]"
				  << std::endl;
		std::cout << "  Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor
				  << std::endl;
		std::cout << std::endl;
	}
}

void MyCudaUtils::PrintMatrix(const char* name, float* matrix, int size)
{
	printf("%s:\n", name);
	for(int i = 0; i < size; i++)
	{
		for(int j = 0; j < size; j++)
		{
			printf("%10.2f ", matrix[i * size + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void MyCudaUtils::RandomFillMatricies(
	float* MatrixA, float* MatrixB, int MatrixSize, int MaxNum, int MinNum)
{

	for(int x = 0; x < MatrixSize; x++)
	{
		for(int y = 0; y < MatrixSize; y++)
		{
			MatrixA[x * MatrixSize + y] = (float)(rand() % (MaxNum - MinNum + 1) + MinNum);
			MatrixB[x * MatrixSize + y] = (float)(rand() % (MaxNum - MinNum + 1) + MinNum);
		}
	}
}
