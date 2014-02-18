/*
 * cudamog.h
 *
 *  Created on: Jan 11, 2013
 *      Author: dbd
 */

#ifndef CUDAMOG_H_
#define CUDAMOG_H_

#include "mogconfig.h"
#include "detectedparameters.h"

//OPENCV 2.2 INCLUDES.....
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;

//NVIDIA Performance Primitives
#include <npp.h>

struct DetectedParameters;
struct cudaArray;
struct MoGConfig;

extern "C" void setPre_sigmaR2(float *h_Kernel, size_t mem_size);

void initialiseAndProcessFirstFrame (MoGConfig          const *config,
                                     DetectedParameters const *parameters,
                                     cudaArray                *input,
                                     cudaArray                *gaussians,
                                     cudaArray                *order,
                                     cudaArray                *output);

void processFrame (MoGConfig          const *config,
                   DetectedParameters const *parameters,
                   float alpha);

//class fipImage;
//class Mat;
template <typename STORAGE_CLASS> class GaussianParametersTemplate;
template <typename STORAGE_CLASS> class SortingParametersTemplate;

template <int COMPONENTS, typename CUDA_VECTOR_T>
class MixtureOfGaussians {

private:
    MixtureOfGaussians ();
    cudaArray *inputArray;
	cudaArray *foregroundColorArray; 
    cudaArray *outputArray;
	cudaArray *finalOutputArray;
	cudaArray *maskArray;
	cudaArray *alphaArray;
    cudaArray *gaussiansArray;
    cudaArray *sortingArray;
	cudaArray *gaussiansAuxArray;
    cudaArray *sortingAuxArray;
    //float alpha;
    const MoGConfig config;
    const DetectedParameters parameters;
    const dim3 blockSize;
    const dim3 gridSize;
    typedef float2 GAUSSIAN_STORAGE_T; 
    typedef float2 SORTING_STORAGE_T;

    unsigned int processedFrames;
    bool initialised;
public:
    MixtureOfGaussians (MoGConfig          const *config,
                        DetectedParameters const *parameters,
						cudaArray				 **colorArray);
	MixtureOfGaussians (MoGConfig          const *config,
                        DetectedParameters const *parameters,
						cudaArray				 **colorArray,
						cudaArray				 **CmapArray);
    void processImage (Mat const *input, Mat *output, uchar *inputColor, uchar *modeloColor, Npp8u *outColor, float weightMin, float sigmaMax);
	//cudaArray* processImage (Mat const *input, Mat *output);
	//void processImage (Mat const *input, Mat *output, cudaArray *outputArray);
	void processImage (Mat const *input, Mat *output, Mat *finalOutput, unsigned short int *inputDepth, unsigned short int *modeloDepth, float a, float b, float c, float deltaSigma, float weightMin, float sigmaMax, int cMin);
	void processImageOnlyDetection (Mat const *input, Mat *output, Mat *finalOutput, unsigned short int *inputDepth, Npp8u *filledOutput, float a, float b, float c, float deltaSigma);
	//void processImage (fipImage const *input, fipImage *output);
    virtual ~MixtureOfGaussians ();
    typedef GaussianParametersTemplate<GAUSSIAN_STORAGE_T> GaussianParameters;
    typedef SortingParametersTemplate<SORTING_STORAGE_T> SortingParameters;
};

#endif /* CUDAMOG_H_ */
