#ifndef AUXILIARKERNELS_H
#define AUXILIARKERNELS_H

#include "stdafx.h"

//#define BLOCK_SIZE 32
#define BLOCK_SIZE 16

extern "C" void recortar(const unsigned short int *inputD, const uchar* inputC, unsigned short int* outputD, uchar* outputC, int IF, int IC, int nl, int nc, int nct, int nch);

//extern "C" void rellenarBeforeFiltering(const unsigned short int *input, const unsigned short int *modelo, uchar* maskForeground, uchar* maskBordersDD, unsigned short int *output, unsigned short int *outputModel, int nl, int nc);
extern "C" void rellenarBeforeFiltering(const unsigned short int *input, const unsigned short int *modelo, uchar* maskForeground, unsigned short int *output, unsigned short int *outputModel, int nl, int nc);

extern "C" void rellenarBordersDepthNoColor(const unsigned short int *input, const unsigned short int *modelo, uchar* maskBordes, uchar* maskDNC, unsigned short int *output, int nl, int nc);
#endif
