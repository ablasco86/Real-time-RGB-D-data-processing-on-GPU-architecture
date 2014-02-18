#ifndef AUXFUNCS_H
#define AUXFUNCS_H

#include "stdafx.h"

#define PI 3.14159265358979323846

void filterInit(float* data, int size, float sigmas);

void filterInit1D(float* data, int size, float sigmas);

void createSobelH(int* data);

void createSobelV(int* data);

void pre_calculation(float* data, int size, float sigmar);

void pre_sigmaR(float* data, int size, float a, float b, float c);

void calcSigma(float mean, float a, float b, float c, float deltaSigma, float *sigmaMin, float *sigma0);

void conversion(const char* fileName, int &filas, int &columnas, Mat &imagen);

void writeVector(const char * str, vector<float> &dataVector);

#endif
