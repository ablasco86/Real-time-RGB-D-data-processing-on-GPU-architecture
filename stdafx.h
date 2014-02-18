// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

//OPENCV 2.2 INCLUDES.....
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;

//MISCELLANEOUS INCLUDES.....
#include <stdio.h>
#include <tchar.h>
#include <iostream>
using namespace std;

/*///local include
#include "kinectLib.h"
#include "kinectLibSave.h"
//#include "lineCommand.h"

//OPEN NUI INCLUDES.....
#include <XnOpenNI.h>
#include <XnLog.h>
#include <XnCppWrapper.h>
#include <XnFPSCalculator.h>
using namespace xn;*/

#include "cuda_runtime.h"

//NVIDIA Performance Primitives
#include <npp.h>

// Matrices are stored in row-major order: 
// M(row, col) = *(M.elements + row * M.width + col) 
typedef struct { 
	unsigned int width; 
	unsigned int height; 
	float* elements; 
} Matrix;

typedef struct { 
	unsigned int width; 
	unsigned int height; 
	int* elements; 
} Matrix2;

// TODO: reference additional headers your program requires here
