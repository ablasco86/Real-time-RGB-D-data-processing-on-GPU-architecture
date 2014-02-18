/*
 * detectedparameters.cpp
 *
 *  Created on: Jan 11, 2013
 *      Author: dbd
 */

#include "detectedparameters.h"

DetectedParametersFactory& DetectedParametersFactory::setWidth (
        unsigned int width)
{
    this->p.width = width;
    return *this;
}

DetectedParametersFactory& DetectedParametersFactory::setHeight (
        unsigned int height)
{
    this->p.height = height;
    return *this;
}

DetectedParameters DetectedParametersFactory::toStruct () const
{
    return this->p;
}
