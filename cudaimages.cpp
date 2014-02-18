/*
 * cudaimages.cpp
 *
 *  Created on: Jan 10, 2013
 *      Author: dbd
 */

//OPENCV 2.2 INCLUDES.....
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
using namespace cv;

#include <cuda_runtime.h>
//#include <FreeImagePlus.h>
#include <cassert>
#include <iostream>
#include "cudaimages.h"
#include "generalcuda.h"


template <typename T>
cudaArray* allocateArrayForImage (unsigned int width, unsigned int height)
{
    cudaArray *array;
    const cudaExtent arrayDims = make_cudaExtent(width, height, 0); // Docs say width should be bytes, but it's actually elements
    const cudaChannelFormatDesc arrayFormat = cudaCreateChannelDesc<T>();
    CUDA_SAFE_CALL (cudaMalloc3DArray (&array, &arrayFormat, arrayDims, cudaArraySurfaceLoadStore));
    return array;
}

template cudaArray* allocateArrayForImage<uchar4> (unsigned int width, unsigned int height);
template cudaArray* allocateArrayForImage<unsigned char> (unsigned int width, unsigned int height);
template cudaArray* allocateArrayForImage<unsigned short> (unsigned int width, unsigned int height);

template <typename T>
bool areEqual(const T& a, const T& b);

template <>
bool areEqual<cudaChannelFormatDesc> (const cudaChannelFormatDesc& a,
                                      const cudaChannelFormatDesc& b)
{
    return (a.f == b.f) && (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);
}



int getImageType (cudaChannelFormatDesc const &format) {
//FREE_IMAGE_TYPE getImageType (cudaChannelFormatDesc const &format) {
    switch (format.f) {
    //case cudaChannelFormatKindFloat: // TODO: Ya se implementará si hace falta
    case cudaChannelFormatKindUnsigned:
        if (   ((8 == format.x) && (0 == format.y) && (0 == format.z) && (0 == format.w))
            || ((8 == format.x) && (8 == format.y) && (0 == format.z) && (0 == format.w))
            || ((8 == format.x) && (8 == format.y) && (8 == format.z) && (8 == format.w))) {
			//return FIT_BITMAP;
            return 8;
        }
		if ((16 == format.x) && (0 == format.y) && (0 == format.z) && (0 == format.w))
			//return FIT_UINT16;
			return 16;
        //return FIT_UNKNOWN;
		return 0;
    default:
		return 0;
        //return FIT_UNKNOWN; // This will make other operations fail
    }
}

unsigned int getBPP (cudaChannelFormatDesc const &format) {
    return static_cast<unsigned int>(format.x + format.y + format.z + format.w);
}

bool areCompatible (const Mat image, cudaArray* array) {
//bool areCompatible (const fipImage& image, cudaArray* array) {
    cudaChannelFormatDesc format;
    cudaExtent extent;
    cudaArrayGetInfo(&format, &extent, NULL, array);
    return (getImageType(format) == image.elemSize1()*8)
		&& (getBPP(format) == image.elemSize()*8)
           //&& areEqual(format, cudaCreateChannelDesc<uchar4>())
		   && (image.cols == extent.width)
           && (image.rows == extent.height);
	/*return (getImageType(format) == image.getImageType())
           && (getBPP(format) == image.getBitsPerPixel())
           //&& areEqual(format, cudaCreateChannelDesc<uchar4>())
           && (image.getWidth() == extent.width)
           && (image.getHeight() == extent.height);*/
}


bool allocateImageForArray (Mat& image, cudaArray *array) {
//bool allocateImageForArray (fipImage& image, cudaArray *array) {
    cudaChannelFormatDesc format;
    cudaExtent extent;
    cudaArrayGetInfo(&format, &extent, NULL, array);
    //return image.setSize(getImageType(format), extent.width, extent.height, getBPP(format));
	image.create(extent.width, extent.height, CV_8UC1);
	return true;
}

void copyImageToArray (const Mat& image, cudaArray* array)
//void copyImageToArray (const fipImage& image, cudaArray* array)
{
    assert (areCompatible(image, array));
    cudaExtent extent;
    cudaArrayGetInfo(NULL, &extent, NULL, array);
    assert (0 == extent.depth); // Ensure this is a 2D array

    /// Copy image to array
    cudaMemcpy3DParms memcpyParams = {0};
    memcpyParams.extent = make_cudaExtent (extent.width, extent.height, 1); // Docs say width should be bytes, but it's actually elements
    memcpyParams.kind   = cudaMemcpyHostToDevice;
	//memcpyParams.srcPtr = make_cudaPitchedPtr (const_cast<unsigned char *>(image.accessPixels ()),
                                               //image.getScanWidth (), extent.width, extent.height);
	memcpyParams.srcPtr = make_cudaPitchedPtr (const_cast<unsigned char *>(image.ptr ()),
                                               (image.elemSize()*image.cols), extent.width, extent.height);
    memcpyParams.dstArray = array;
    CUDA_SAFE_CALL (cudaMemcpy3D (&memcpyParams));
}

void copyImageFromArray (Mat& image, cudaArray* array) {
//void copyImageFromArray (fipImage& image, cudaArray* array) {
    if (!areCompatible(image, array)) {
        assert(allocateImageForArray(image, array));
    }
    cudaExtent extent;
    cudaArrayGetInfo(NULL, &extent, NULL, array);
    assert (0 == extent.depth); // Ensure this is a 2D array

    /// Copy image from array
    cudaMemcpy3DParms memcpyParams = {0};
    memcpyParams.extent = make_cudaExtent (extent.width, extent.height, 1); // Docs say width should be bytes, but it's actually elements
    memcpyParams.kind   = cudaMemcpyDeviceToHost;
	memcpyParams.dstPtr = make_cudaPitchedPtr (const_cast<unsigned char *>(image.ptr ()),
											   (image.elemSize()*image.cols), extent.width, extent.height);
    //memcpyParams.dstPtr = make_cudaPitchedPtr (const_cast<unsigned char *>(image.accessPixels ()),
                                               //image.getScanWidth (), extent.width, extent.height);
    memcpyParams.srcArray = array;
    CUDA_SAFE_CALL (cudaMemcpy3D (&memcpyParams));
}





