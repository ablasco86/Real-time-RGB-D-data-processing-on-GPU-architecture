#ifndef MEDIANFILTER_H
#define MEDIANFILTER_H

#include "stdafx.h"

//#define BLOCK_W 4
//#define BLOCK_H 4
#define BLOCK_SIZE 4
#define FILTER_SIZE 13
//#define PI 3.14159265358979323846

//extern "C" void imageMedianFilter(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, float sigmar, float thc);
extern "C" void imageMedianFilter(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc);

extern "C" void imageMedianFilterAprox(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc);

extern "C" void imageMedianFilterH(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc);

extern "C" void imageMedianFilterV(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc);

#endif
