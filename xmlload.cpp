#include "xmlload.h"

/* Leer un fichero XML
 *
 * Carga un fichero de tipo XML donde se almacenan los valores de los parámetros que usaremos en cada prueba 
 * del sistema.
 *
 */
float ReturnFloat (TiXmlDocument &doc, const char* nameVariable)
{
	TiXmlNode* node = 0;
	float valor = 0.0;
	String variable;

	node = doc.RootElement();
	node = node->FirstChild("float");
	node = node->FirstChild(nameVariable);
	node = node->FirstChild();
	variable = node->Value();

	valor = atof(variable.c_str());

	return valor;

}

int ReturnInt (TiXmlDocument &doc, const char* nameVariable)
{
	TiXmlNode* node = 0;
	int valor = 0;
	String variable;

	node = doc.RootElement();
	node = node->FirstChild("int");
	node = node->FirstChild(nameVariable);
	node = node->FirstChild();
	variable = node->Value();

	valor = atoi(variable.c_str());

	return valor;

}

String ReturnString(TiXmlDocument &doc, const char* nameVariable)
{

	TiXmlNode* node = 0;
	String valor;

	node = doc.RootElement();
	node = node->FirstChild("string");
	node = node->FirstChild(nameVariable);
	node = node->FirstChild();

	valor = node->Value();

	return valor;

}

QString ReturnQString(TiXmlDocument &doc, const char* nameVariable)
{

	TiXmlNode* node = 0;
	QString valor;

	node = doc.RootElement();
	node = node->FirstChild("qstring");
	node = node->FirstChild(nameVariable);
	node = node->FirstChild();

	valor = node->Value();

	return valor;

}
