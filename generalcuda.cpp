/*
 * generalcuda.cpp
 *
 *  Created on: Jan 10, 2013
 *      Author: dbd
 */
#include <cuda_runtime.h>
#include "generalcuda.h"

bool setupFirstCuda2xDevice () {
    int numberOfDevices = 0;
    if (cudaSuccess != cudaGetDeviceCount (&numberOfDevices)) {
        return false;
    }
    for (int d = 0; d < numberOfDevices; ++d) {
        cudaDeviceProp properties;
        if (cudaSuccess != cudaGetDeviceProperties (&properties, d)) {
            continue;
        }
        if ((2 == properties.major) && (cudaSuccess == cudaSetDevice(d))) {
            return true;
        }
    }
    return false;
}
