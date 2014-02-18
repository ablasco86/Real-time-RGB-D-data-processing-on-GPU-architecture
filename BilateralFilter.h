#ifndef BILATERALFILTER_H
#define BILATERALFILTER_H

#include "stdafx.h"
#include "detectedparameters.h"

#define BLOCK_SIZE 16
#define FILTER_SIZE 13
#define PI 3.14159265358979323846

extern "C" void setDistanceMask(float *h_Kernel, size_t mem_size);

extern "C" void setDistanceMask1D(float *h_Kernel, size_t mem_size);

extern "C" void setPre_calculation(float *h_Kernel, size_t mem_size);

extern "C" void setPre_sigmaR(float *h_Kernel, size_t mem_size);

extern "C" void setTextureRellenado(DetectedParameters const *parameters, cudaArray **CmapArray);

extern "C" void imageBilateralFilter(const unsigned short int* input, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, float a, float b, float c);

extern "C" void imageBilateralFilterH(const unsigned short int* input, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, float a, float b, float c);

extern "C" void imageBilateralFilterV(const unsigned short int* input, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, float a, float b, float c);

extern "C" void imageRellenar(const Matrix F, const unsigned short int* input, const uchar* color, const uchar* modeloColor, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int sigmar, float porcen, int cMin);

extern "C" void imageRellenarH(const Matrix F, const unsigned short int* input, const uchar* color, const uchar* modeloColor, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int sigmar, float porcen, int cMin);

extern "C" void imageRellenarV(const Matrix F, const unsigned short int* input, const uchar* color, const uchar* modeloColor, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int sigmar, float porcen, int cMin);

#endif
