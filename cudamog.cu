#include <texture_types.h>
#include <texture_fetch_functions.h>
#include <surface_types.h>
#include <surface_functions.h>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include "cudamog.h"
#include "generalcuda.h"
#include "mogconfig.h"
#include "detectedparameters.h"
#include "cudaimages.h"


template <typename CUDA_VECTOR>
class VectorWrapper : public CUDA_VECTOR {
public:
    __device__ void read (surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer) {
        surf2DLayeredread<CUDA_VECTOR> (this, surf, sizeof(CUDA_VECTOR) * x, y, layer);
    }
    __device__ void write (surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer) const {
        surf2DLayeredwrite<CUDA_VECTOR> (*this, surf, sizeof(CUDA_VECTOR) * x, y, layer);
    }
    __device__ VectorWrapper (surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer) {
        this->read(surf, x, y, layer);
    }
    __device__ VectorWrapper () {}
};

template <typename STORAGE_CLASS>
class GaussianParametersTemplate : public VectorWrapper<STORAGE_CLASS> {
public:
    __device__ void setMean (float mean) { this->x = mean; }
    __device__ void setStdDev (float stdDev) {this->y = stdDev; }
    __device__ float getMean () const { return this->x; }
    __device__ float getStdDev () const { return this->y; }
	__device__ GaussianParametersTemplate (float mean, float stdDev) { this->setMean(mean); this->setStdDev(stdDev);}
    __device__ GaussianParametersTemplate (surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer) {
        this->read(surf, x, y, layer);
    }
    __device__ GaussianParametersTemplate () {}
};

template <typename STORAGE_CLASS>
class SortingParametersTemplate : public VectorWrapper<STORAGE_CLASS> {
public:
    __device__ float getWeight () const { return this->x; }
    __device__ float getSortingKey () const { return this->y; }
	//__device__ float getStdDevMin () const { return this->z; }
    __device__ void setWeight (float weight) { this->x = weight; }
    __device__ void setSortingKey (float sortingKey) { this->y = sortingKey; }
	//__device__ void setStdDevMin (float stdDevMin) {this->z = stdDevMin; }
    __device__ void updatePositiveMatch (float alpha, float stdDevSum, float sigmaInit) {
        this->setWeight((1.0f - alpha) * this->getWeight () + alpha);
        this->setSortingKey(this->getWeight () / (stdDevSum/sigmaInit));
    }
    __device__ void updateNegativeMatch (float alpha) {
        this->setWeight((1.0f - alpha) * this->getWeight ());
        this->setSortingKey((1.0f - alpha) * this->getSortingKey ());
    }
    //__device__ SortingParametersTemplate (float weight, float sortingKey, float stdDevMin) { this->setWeight(weight); this->setSortingKey(sortingKey); this->setStdDevMin(stdDevMin);}
	__device__ SortingParametersTemplate (float weight, float sortingKey) { this->setWeight(weight); this->setSortingKey(sortingKey);}
    __device__ SortingParametersTemplate () {}
    __device__ SortingParametersTemplate (surface<void, cudaSurfaceType2DLayered> surf, int x, int y, int layer) {
        this->read(surf, x, y, layer);
    }
};


//template <int COMPONENTS, typename SOURCE_CHANNEL_TYPE>
//__device__ void initializePixel (float pixel[COMPONENTS]);
template <typename SOURCE_CHANNEL_TYPE>
__device__ void initializePixel (float pixel[]);

template <int COMPONENTS, typename CUDA_VECTOR_T>
__device__ bool matchesMode(const MoGConfig &config,
                            const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::GaussianParameters gaussianParameters[COMPONENTS],
                            const float pixel[COMPONENTS]) {
    float sum = 0.0f;
    for (int i = 0; i < COMPONENTS; ++i) {
        const float diff = pixel[i] - gaussianParameters[i].getMean();
        const float ratio = diff / gaussianParameters[i].getStdDev();
        sum += (ratio * ratio);
    }
    return (sum < (config.lambda * config.lambda));
}

template <int COMPONENTS, typename CUDA_VECTOR_T>
MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::MixtureOfGaussians (MoGConfig          const *config,
                                                                   DetectedParameters const *parameters)
    : config (*config),
      parameters (*parameters),
      initialised (false),
      processedFrames (0),
      blockSize (config->blockWidth, config->blockHeight),
      gridSize (((parameters->width + blockSize.x - 1) / blockSize.x),
                ((parameters->height + blockSize.y - 1) / blockSize.y))
{
    this->inputArray = allocateArrayForImage<CUDA_VECTOR_T>(this->parameters.width, this->parameters.height);

    inputTexture.addressMode[0] = cudaAddressModeBorder;
    inputTexture.addressMode[1] = cudaAddressModeBorder;
    inputTexture.filterMode = cudaFilterModePoint;
    inputTexture.normalized = false;
    CUDA_SAFE_CALL (cudaBindTextureToArray(inputTexture, this->inputArray));

    this->outputArray = allocateArrayForImage<unsigned char>(this->parameters.width, this->parameters.height);
    CUDA_SAFE_CALL (cudaBindSurfaceToArray (outputSurface, this->outputArray));

	this->maskArray = allocateArrayForImage<unsigned char>(this->parameters.width, this->parameters.height);
    CUDA_SAFE_CALL (cudaBindSurfaceToArray (maskSurface, this->maskArray));

	this->alphaArray = allocateArrayForImage<unsigned char>(this->parameters.width, this->parameters.height);
    CUDA_SAFE_CALL (cudaBindSurfaceToArray (alphaSurface, this->alphaArray));

    const cudaExtent gaussiansArrayDims = make_cudaExtent(this->parameters.width, this->parameters.height, COMPONENTS * this->config.k); // Docs say width should be bytes, but it's actually elements
    const cudaChannelFormatDesc gaussiansArrayFormat = cudaCreateChannelDesc<GAUSSIAN_STORAGE_T>();
    CUDA_SAFE_CALL (cudaMalloc3DArray (&this->gaussiansArray, &gaussiansArrayFormat, gaussiansArrayDims, cudaArrayLayered | cudaArraySurfaceLoadStore));
    CUDA_SAFE_CALL (cudaBindSurfaceToArray (gaussiansSurface, this->gaussiansArray));


    const cudaExtent sortingArrayDims = make_cudaExtent(this->parameters.width, this->parameters.height, this->config.k); // Docs say width should be bytes, but it's actually elements
    const cudaChannelFormatDesc sortingArrayFormat = cudaCreateChannelDesc<SORTING_STORAGE_T> ();
    CUDA_SAFE_CALL (cudaMalloc3DArray (&this->sortingArray, &sortingArrayFormat, sortingArrayDims, cudaArrayLayered | cudaArraySurfaceLoadStore));
    CUDA_SAFE_CALL (cudaBindSurfaceToArray (sortingSurface, this->sortingArray));
}

template<int COMPONENTS, typename CUDA_VECTOR_T>
MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::~MixtureOfGaussians ()
{
    CUDA_SAFE_CALL (cudaFreeArray(this->inputArray));
    CUDA_SAFE_CALL (cudaFreeArray(this->outputArray));
	CUDA_SAFE_CALL (cudaFreeArray(this->maskArray));
	CUDA_SAFE_CALL (cudaFreeArray(this->alphaArray));
    CUDA_SAFE_CALL (cudaFreeArray(this->gaussiansArray));
    CUDA_SAFE_CALL (cudaFreeArray(this->sortingArray));
}

template <int COMPONENTS, typename CUDA_VECTOR_T>
__global__ void mogInitialiseKernel (MoGConfig           config,
                                     DetectedParameters  parameters) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((parameters.width <= x) || (parameters.height <= y)) { return; }

	float a = 0.00000235;
	float b = 0.00055;
	float c = 2.3;

    // TODO: Hacer algo para cuando hay sólo un canal (depth)
    
	float pixel[COMPONENTS];
	
    initializePixel<CUDA_VECTOR_T>(pixel);

	if ((COMPONENTS == 1) && (pixel[0] == 0)){
		// Not initialized
		const unsigned char initialized = 0;
		surf2Dwrite<unsigned char> (initialized, maskSurface, sizeof (unsigned char) * x, y);
		// This first frame is unconditionally classified as background.
		const unsigned char background = 0;
		surf2Dwrite<unsigned char> (background, outputSurface, sizeof (unsigned char) * x, y);
		return;
	}

	if (COMPONENTS == 1){
		unsigned char processedPixels = 1;
		surf2Dwrite<unsigned char> (processedPixels, alphaSurface, sizeof (unsigned char) * x, y);
	}

	//const unsigned char initialized = 255;
	// Initialized but not stable
	const unsigned char initialized = 128;
	surf2Dwrite<unsigned char> (initialized, maskSurface, sizeof (unsigned char) * x, y);

    // We initialize all the modes with the same parameters using this first
    // image. Since they are tried in order, they will slowly evolve to
    // reflect actual modes of the sequence.
	float sigma0 = config.sigma_0;
	float sigmaMin = config.sigma_min;
    const float initialWeight  = 1.0f / static_cast<float>(config.k);
    float initialSortkey = initialWeight / (static_cast<float>(COMPONENTS) * sigma0); //config.sigma_0
	if (COMPONENTS == 1){
		sigmaMin = c+b*(pixel[0])+a*(pixel[0])*(pixel[0]);
		sigma0 = 2.5 * sigmaMin;
		initialSortkey *= sigma0;
	}
    for (int mode = 0; mode < config.k; ++mode) {
        for (int component = 0; component < COMPONENTS; ++component) {
            const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::GaussianParameters gaussianParameters (pixel[component], sigma0); //config.sigma_0
            gaussianParameters.write(gaussiansSurface, x, y, COMPONENTS * mode + component);
        }
        const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::SortingParameters sortingParameters (initialWeight, initialSortkey);
        sortingParameters.write(sortingSurface, x, y, mode);
    }

    // This first frame is unconditionally classified as background.
    const unsigned char background = 0;
    surf2Dwrite<unsigned char> (background, outputSurface, sizeof (unsigned char) * x, y);
}




template <int COMPONENTS, typename CUDA_VECTOR_T>
__global__ void mogProcessKernel (float alpha,
                                  MoGConfig           config,
                                  DetectedParameters  parameters)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((parameters.width <= x) || (parameters.height <= y)) { return; }

	float a = 0.00000235;
	float b = 0.00055;
	float c = 2.3;

    float pixel[COMPONENTS];
    initializePixel<CUDA_VECTOR_T> (pixel);

	if ((COMPONENTS == 1) && (pixel[0] == 0)){ 
		const unsigned char background = 0;
		surf2Dwrite<unsigned char> (background, outputSurface, sizeof (unsigned char) * x, y);	
		return; 
	}

	unsigned char initialized;

	surf2Dread<unsigned char> (&initialized, maskSurface, sizeof (unsigned char) *x, y);

	float sigmaMin = config.sigma_min;

	if (initialized == 0){

		unsigned char processedPixels = 1;
		surf2Dwrite<unsigned char> (processedPixels, alphaSurface, sizeof (unsigned char) * x, y);

		// Initialized but not stable
		initialized = 128;
		surf2Dwrite<unsigned char> (initialized, maskSurface, sizeof (unsigned char) * x, y);

		// We initialize all the modes with the same parameters using this first
		// image. Since they are tried in order, they will slowly evolve to
		// reflect actual modes of the sequence.
		const float initialWeight  = 1.0f / static_cast<float>(config.k);
		float initialSortkey = initialWeight / (static_cast<float>(COMPONENTS) * config.sigma_0);
		float sigma0 = config.sigma_0;
		if (COMPONENTS == 1){
			sigmaMin = c+b*(pixel[0])+a*(pixel[0])*(pixel[0]);
			sigma0 = sigmaMin * 2.5;
			initialSortkey *= sigma0;
		}
		for (int mode = 0; mode < config.k; ++mode) {
			for (int component = 0; component < COMPONENTS; ++component) {
				const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::GaussianParameters gaussianParameters (pixel[component], sigma0);
				gaussianParameters.write(gaussiansSurface, x, y, COMPONENTS * mode + component);
			}
			const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::SortingParameters sortingParameters (initialWeight, initialSortkey);
			sortingParameters.write(sortingSurface, x, y, mode);
		}

		// This first pixel with valid data is unconditionally classified as background.
		const unsigned char background = 0;
		surf2Dwrite<unsigned char> (background, outputSurface, sizeof (unsigned char) * x, y);

		return;
	}

	if (COMPONENTS == 1){
		unsigned char processedPixels;
		surf2Dread<unsigned char> (&processedPixels, alphaSurface, sizeof (unsigned char) *x, y);
		if (processedPixels < static_cast<unsigned char>(1.0f/config.alpha_min))
			processedPixels++;
		alpha = 1.0f / static_cast<float>(processedPixels);
		surf2Dwrite<unsigned char> (processedPixels, alphaSurface, sizeof (unsigned char) * x, y);
	}

    int matchingMode = -1; // Not valid match
    for (int mode = 0; mode < config.k; ++mode) {
        typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::SortingParameters sortingParameters (sortingSurface, x, y, mode);
        typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::GaussianParameters gaussianParameters[COMPONENTS];
        for (int component = 0; component < COMPONENTS; ++component) {
            gaussianParameters[component].read(gaussiansSurface, x, y, COMPONENTS * mode + component);
        }
        // Only the first matching mode is considered
        if ((0 > matchingMode) && (matchesMode<COMPONENTS, CUDA_VECTOR_T>(config, gaussianParameters, pixel))) {
            matchingMode = mode;
            //const float SQRT_2PI = 2.506628275f;
            float rho = alpha;
            // TODO : Preguntar qué pichas hacer con la normalización, aunque tiene pinta de que normFactor = 1,
            // dado que rho debe estar entre 0 y 1 y alpha también lo está.
            //float normFactor = SQRT_2PI;
            for (int component = 0; component < COMPONENTS; ++component) {
                const float arg = (pixel[component] - gaussianParameters[component].getMean())
                                  / gaussianParameters[component].getStdDev();
                // TODO: Cambiar esta exponencial por la tabla
                rho *= __expf(-0.5f * arg * arg);
                // TODO : Preguntar qué pichas hacer con la normalización
                //normFactor *= gaussianParameters[component].getStdDev();
            }
            //rho /= normFactor;
            float stdDevSum = 0.0f;
			//float sigmaMin = config.sigma_min;
			//float sigma0 = config.sigma_0;
			float sigma0 = 1; //Para que no me afecte en el color
			if (COMPONENTS ==1){
				sigmaMin = c+b*(gaussianParameters[0].getMean())+a*(gaussianParameters[0].getMean())*(gaussianParameters[0].getMean());
				sigma0 = sigmaMin * 2.5;
			}
            for (int component = 0; component < COMPONENTS; ++component) {
                gaussianParameters[component].setMean((1.0f - rho) * gaussianParameters[component].getMean()
                                                      + rho * pixel[component]);
                // TODO : Cerciorarse de que primero se actualiza la media y luego se actualiza la stdDev
//                gaussianParameters[component].setStdDev((1.0f - rho) * gaussianParameters[component].getStdDev()
//                                                        + rho * fabsf (pixel[component] - gaussianParameters[component].getMean()));
                gaussianParameters[component].setStdDev(__fsqrt_rn((1.0f - rho) * gaussianParameters[component].getStdDev() * gaussianParameters[component].getStdDev()
                                                        + rho * (pixel[component] - gaussianParameters[component].getMean()) * (pixel[component] - gaussianParameters[component].getMean())));
                gaussianParameters[component].setStdDev((sigmaMin > gaussianParameters[component].getStdDev())
                                                        ? sigmaMin : gaussianParameters[component].getStdDev());
                stdDevSum += gaussianParameters[component].getStdDev ();
                gaussianParameters[component].write(gaussiansSurface, x, y, COMPONENTS * mode + component);
            }
            sortingParameters.updatePositiveMatch(alpha, stdDevSum, sigma0);
        } else {
            sortingParameters.updateNegativeMatch(alpha);
        }
        sortingParameters.write(sortingSurface, x, y, mode);
    }

   ((-1 == matchingMode) && (300 == x) && (240 == y)) ? printf ("No hay cerillas\n") : 0;
    // If the match is in the first mode we do not need to reorder the modes
    if (0 < matchingMode) {
        typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::GaussianParameters matchingGaussian[COMPONENTS];
        for (int c = 0; c < COMPONENTS; ++c) {
            matchingGaussian[c].read(gaussiansSurface, x, y, COMPONENTS * matchingMode + c);
        }
        const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::SortingParameters matchingSorting (sortingSurface, x, y, matchingMode);

        for (int mode = (matchingMode - 1); mode >= 0; --mode) {
            {
                const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::SortingParameters otherSorting (sortingSurface, x, y, mode);
                if (otherSorting.getSortingKey() >= matchingSorting.getSortingKey()) {
                    matchingSorting.write(sortingSurface, x, y, mode + 1);
                    for (int c = 0; c < COMPONENTS; ++c) {
                        matchingGaussian[c].write(gaussiansSurface, x, y, COMPONENTS * (mode + 1) + c);
                    }
                    matchingMode = mode + 1;
                    break;
                }
                otherSorting.write(sortingSurface, x, y, mode + 1);
            }
            for (int c = 0; c < COMPONENTS; ++c) {
                const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::GaussianParameters otherGaussian (gaussiansSurface, x, y, COMPONENTS * mode + c);
                otherGaussian.write(gaussiansSurface, x, y, COMPONENTS * (mode + 1) + c);
            }
        }
    }

    // If there was no match, we must insert a new mode in place of the
    // last mode
    if (-1 == matchingMode) {
        matchingMode = config.k - 1;
        const float initialWeight = config.w_0;
        float initialSortkey = initialWeight / (COMPONENTS * config.sigma_0);
		float sigma0 = config.sigma_0;
		//float sigmaMin = config.sigma_min;
		if (COMPONENTS == 1){
			sigmaMin = c+b*(pixel[0])+a*(pixel[0])*(pixel[0]);
			sigma0 = sigmaMin * 2.5;
			initialSortkey *= sigma0;
		}
        for (int component = 0; component < COMPONENTS; ++component) {
            const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::GaussianParameters gaussianParameters (pixel[component], sigma0);
            //((x == 200) && (y == 200)) ? printf ("  sigma_0: %f\n", getStdDev(gaussianParameters)) : 0;
            gaussianParameters.write(gaussiansSurface, x, y, COMPONENTS * matchingMode + component);
        }
        const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::SortingParameters sortingParameters (initialWeight, initialSortkey);
        sortingParameters.write(sortingSurface, x, y, matchingMode);
    }

	// If initialized but not stable, we must check if it becomes stable (only Mode 0)
	if (initialized == 128){
		int mode = 0;
		typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::SortingParameters sortingParameters (sortingSurface, x, y, mode);
        typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::GaussianParameters gaussianParameters[COMPONENTS];
        for (int component = 0; component < COMPONENTS; ++component) {
            gaussianParameters[component].read(gaussiansSurface, x, y, COMPONENTS * mode + component);
        }

		float stdDevSum = 0.0f;
		//float stdDevSumMin = 0.0f;
        for (int component = 0; component < COMPONENTS; ++component){
			stdDevSum += gaussianParameters[component].getStdDev ();
			//stdDevSumMin += sortingParameters.getStdDevMin();
		}
		const float weight = sortingParameters.getWeight();

		if ((weight > 0.6) && (stdDevSum < 1.1*COMPONENTS*sigmaMin)){
			initialized = 255;
			surf2Dwrite<unsigned char> (initialized, maskSurface, sizeof (unsigned char) * x, y);
		}

	}

    // Now we must renormalize the weights of the modes before classifying the
    // pixel as fore- or background.
    float weightNormalization = 0.0f;
    float accumulatedWeightBeforeMatch = 0.0f;
    bool pixelIsBackground;
    for (int mode = 0; mode < config.k; ++mode) {
        const typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::SortingParameters sortingParameters (sortingSurface, x, y, mode);
        weightNormalization += sortingParameters.getWeight();
    }
    for (int mode = 0; mode < config.k; ++mode) {
        typename MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::SortingParameters sortingParameters (sortingSurface, x, y, mode);
        const float newWeight = sortingParameters.getWeight() / weightNormalization;
//        accumulatedWeightBeforeMatch += newWeight;
//        sortingParameters.setWeight(newWeight);
//        sortingParameters.setSortingKey(sortingParameters.getSortingKey() / weightNormalization);
//        sortingParameters.write (sortingSurface, x, y, mode);
        if (mode == matchingMode) {
          pixelIsBackground = (accumulatedWeightBeforeMatch < config.thresh) ? true : false;
        }
        accumulatedWeightBeforeMatch += newWeight;
        sortingParameters.setWeight(newWeight);
        sortingParameters.setSortingKey(sortingParameters.getSortingKey() / weightNormalization);
        sortingParameters.write (sortingSurface, x, y, mode);
    }

    ((x == 300) && (y == 240)) ? printf ("%d\n", matchingMode) : 0;
    if (pixelIsBackground || initialized < 255) {
        const unsigned char finalPixel = 0;
        surf2Dwrite<unsigned char> (finalPixel, outputSurface, sizeof(unsigned char) * x, y);
    } else {
        const unsigned char finalPixel = 255;
        surf2Dwrite<unsigned char> (finalPixel, outputSurface, sizeof(unsigned char) * x, y);
    }
}







template<int COMPONENTS, typename CUDA_VECTOR_T>
void MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::processImage (Mat const *input, Mat *output)
//void MixtureOfGaussians<COMPONENTS, CUDA_VECTOR_T>::processImage (fipImage const *input, fipImage* output)
{
    ++this->processedFrames;
    float alpha = 1.0f / static_cast<float>(this->processedFrames);
    if (alpha < this->config.alpha_min) {
        alpha = this->config.alpha_min;
    }
    //alpha = 0.0f;
    copyImageToArray(*input, this->inputArray);
    if (!this->initialised) {
        this->initialised = true;
        mogInitialiseKernel<COMPONENTS, CUDA_VECTOR_T>
                           <<<this->gridSize, this->blockSize>>>(this->config,
                                                                 this->parameters);
        CUDA_SAFE_CALL (cudaGetLastError());
    } else {
        mogProcessKernel<COMPONENTS, CUDA_VECTOR_T>
                        <<<this->gridSize, this->blockSize>>>(alpha,
                                                              this->config,
                                                              this->parameters);
    }
    copyImageFromArray(*output, this->outputArray);
}









