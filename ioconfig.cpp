#include "ioconfig.h"

IOConfig& IOConfig::setInputDir  (const QString& id) {
    this->inputDir.setPath (id);
    return *this;
}

IOConfig& IOConfig::setOutputDir (const QString& od) {
    this->outputDir.setPath (od);
    return *this;
}

IOConfig& IOConfig::setFilenameTemplate (const QString& fnt) {
    this->filenameTemplate = fnt;
    return *this;
}

IOConfig& IOConfig::setFirstFrameNumber (int ffn) {
    this->first = ffn;
    return *this;
}

IOConfig& IOConfig::setLastFrameNumber (int lfn) {
    this->last = lfn;
    return *this;
}


IOConfig::IOConfig (const IOConfig& other) :
    inputDir                  (other.inputDir),
    outputDir                 (other.outputDir),
    filenameTemplate          (other.filenameTemplate),
    first                     (other.first),
    last                      (other.last)
{
    // Not much left to do.
}
