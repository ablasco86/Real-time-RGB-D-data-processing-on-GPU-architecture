/*
 * generalcuda.h
 *
 *  Created on: Jan 10, 2013
 *      Author: dbd
 */

#ifndef GENERALCUDA_H_
#define GENERALCUDA_H_

#include <cstdio>
#include <cstdlib>

bool setupFirstCuda2xDevice ();

#define CUDA_SAFE_CALL_NO_SYNC( call) {                                      \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        std::exit(EXIT_FAILURE);                                             \
    } }

#define CUDA_SAFE_CALL( call)     CUDA_SAFE_CALL_NO_SYNC(call);

#endif /* GENERALCUDA_H_ */
