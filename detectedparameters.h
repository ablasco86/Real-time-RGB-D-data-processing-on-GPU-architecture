/*
 * detectedparameters.h
 *
 *  Created on: Jan 11, 2013
 *      Author: dbd
 */

#ifndef DETECTEDPARAMETERS_H_
#define DETECTEDPARAMETERS_H_

#include <vector_types.h>

struct DetectedParameters {
    int width;  // Signed int because CUDA prefers this
    int height; // Signed int because CUDA prefers this
};

class DetectedParametersFactory {
public:
    DetectedParametersFactory () {};
    DetectedParametersFactory& setWidth  (unsigned int width);
    DetectedParametersFactory& setHeight (unsigned int height);
    DetectedParameters toStruct() const;
private:
    DetectedParameters p;
};

#endif /* DETECTEDPARAMETERS_H_ */
