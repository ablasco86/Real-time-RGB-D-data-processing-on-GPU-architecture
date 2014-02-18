#ifndef XMLLOAD_H
#define XMLLOAD_H

//#include "stdafx.h"

#include <tinystr.h>
#include <tinyxml.h>

#include <QtCore/QString>
#include <QtCore/QDir>

float ReturnFloat (TiXmlDocument &doc, const char* nameVariable);

int ReturnInt (TiXmlDocument &doc, const char* nameVariable);

String ReturnString(TiXmlDocument &doc, const char* nameVariable);

QString ReturnQString(TiXmlDocument &doc, const char* nameVariable);

#endif
