#pragma once
namespace MyCudaUtils
{
void OutputGpuInformation();

void PrintMatrix(const char* name, float* matrix, int size);

void RandomFillMatricies(
	float* MatrixA, float* MatrixB, int MatrixSize, int MaxNum, int MinNum);
} // namespace MyCudaUtils
