#ifndef BORDERSDETECT_H
#define BORDERSDETECT_H

#include "stdafx.h"

#define BLOCK_SIZE 16
#define SOBEL_SIZE 3

//extern "C" void setSobelMask(float *h_KernelH, float *h_KernelV, size_t mem_size);
extern "C" void setSobelMask(int *h_KernelH, int *h_KernelV, size_t mem_size);

extern "C" void detectarBordes(const Matrix2 H, const Matrix2 V, const unsigned short int* input, uchar* color, uchar* output, int nl, int nc, int nch, int tam, int thd, int thc);

#endif
