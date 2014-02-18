/*
 * cudaimages.h
 *
 *  Created on: Jan 10, 2013
 *      Author: dbd
 */

#ifndef CUDAIMAGES_H_
#define CUDAIMAGES_H_

struct cudaArray;
//class  fipImage;
//class Mat;

template <typename T>
cudaArray* allocateArrayForImage (unsigned int width, unsigned int height);

//void copyImageToArray (fipImage const& image, cudaArray* array);
void copyImageToArray (const Mat& image, cudaArray* array);
void copyImageFromArray (Mat & image, cudaArray* array);
//void copyImageFromArray (Mat  image, cudaArray* array);

#endif /* CUDAIMAGES_H_ */
