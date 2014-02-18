#ifndef IOCONFIG_H
#define IOCONFIG_H

#include <QtCore/QString>
#include <QtCore/QDir>

class IOConfig {
public:
    IOConfig (void) {}
    IOConfig& setInputDir  (const QString& id);
    IOConfig& setOutputDir (const QString& od);
    IOConfig& setFilenameTemplate (const QString& fnt);
    IOConfig& setFirstFrameNumber (int ffn);
    IOConfig& setLastFrameNumber (int lfn);

    IOConfig (const IOConfig& other);

    QDir inputDir;
    QDir outputDir;
    QString filenameTemplate;
    int first;
    int last;
};



#endif // IOCONFIG_H
