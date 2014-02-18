#include "xmlcreate.h"

/* Creación de un fichero XML
 *
 * Crea un fichero de tipo XML para almacenar los valores de los parámetros que usaremos en cada prueba 
 * del sistema.
 *
 */
void CreateXMLFile(const char* pFilename)
{

	//Creamos un documento de la version XML 1.0
	TiXmlDocument doc;
	TiXmlElement* msg;
	TiXmlElement* root;
	TiXmlElement* volume;
	TiXmlDeclaration * decl = new TiXmlDeclaration( "1.0", "", "" );
	doc.LinkEndChild( decl );   //No terminamos el nodo de la version hasta que no finalice su nodo hijo

	//creamos un hijo de doc
	root = new TiXmlElement( "parametrosSistema" );

	//creamos un hijo de root nombrado float
	volume = new TiXmlElement( "float" );

	//creamos un hijo de flotante, que es el valor de sigma espacial
	msg = new TiXmlElement( "sigmaS" );
	msg->LinkEndChild( new TiXmlText( "2.5" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es el valor de sigma para el rango de color
	msg = new TiXmlElement( "sigmaCR" );
	msg->LinkEndChild( new TiXmlText( "15.3" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es el valor del umbral para la máscara de bordes en el rango del color
	msg = new TiXmlElement( "thFiltroM" );
	msg->LinkEndChild( new TiXmlText( "0.4" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es el valor de a de la ecuación del ruido
	msg = new TiXmlElement( "a" );
	msg->LinkEndChild( new TiXmlText( "0.00000235"));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es el valor de b de la ecuación del ruido
	msg = new TiXmlElement( "b" );
	msg->LinkEndChild( new TiXmlText( "0.00055"));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es el valor de c de la ecuación del ruido
	msg = new TiXmlElement( "c" );
	msg->LinkEndChild( new TiXmlText( "2.3"));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es el porcentaje de confianza para rellenar
	msg = new TiXmlElement("porcentaje");
	msg->LinkEndChild( new TiXmlText( "0.5" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es desviación estándar inicial para la componente de luminancia
	msg = new TiXmlElement("sigma_0_Lum");
	msg->LinkEndChild( new TiXmlText( "25.5" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es desviación estándar inicial para la componente de luminancia
	msg = new TiXmlElement("sigma_0_Cab");
	msg->LinkEndChild( new TiXmlText( "10" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es desviación estándar inicial para el depth
	msg = new TiXmlElement("sigma_0_depth");
	msg->LinkEndChild( new TiXmlText( "259.2" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es el peso inicial cuando creamos un nuevo modo
	msg = new TiXmlElement("weight_0");
	msg->LinkEndChild( new TiXmlText( "0.05" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es la distancia máxima para hacer match en una gausiana existente
	msg = new TiXmlElement("lambda");
	msg->LinkEndChild( new TiXmlText( "2.5" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es el alpha mínimo que puede alcanzar el sistema
	msg = new TiXmlElement("alpha_min");
	msg->LinkEndChild( new TiXmlText( "0.005" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es el umbral para detectar objetos móviles
	msg = new TiXmlElement("threshold_detect");
	msg->LinkEndChild( new TiXmlText( "0.7" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es desviación estándar mínima para la componente de luminancia
	msg = new TiXmlElement("sigma_min_Lum");
	msg->LinkEndChild( new TiXmlText( "10.2" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es desviación estándar mínima para la componente de luminancia
	msg = new TiXmlElement("sigma_min_Cab");
	msg->LinkEndChild( new TiXmlText( "4" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de flotante, que es desviación estándar mínima para el depth
	msg = new TiXmlElement("sigma_min_depth");
	msg->LinkEndChild( new TiXmlText( "103.68" ));
	volume->LinkEndChild( msg );

	//terminamos el volumen float
	root->LinkEndChild( volume );

	//creamos otro volume de tipo entero
	volume = new TiXmlElement("int");

	//creamos un hijo de entero, que es el tamaño del filtro
	msg = new TiXmlElement("filterSize");
	msg->LinkEndChild( new TiXmlText( "7" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el valor de sigma en el rango
	msg = new TiXmlElement("sigmaR");
	msg->LinkEndChild( new TiXmlText( "30" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el inicio de filas para recortar
	msg = new TiXmlElement("initFilas");
	msg->LinkEndChild( new TiXmlText( "38" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el inicio de columnas para recortar
	msg = new TiXmlElement("initColumnas");
	msg->LinkEndChild( new TiXmlText( "28" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el número de filas de la imagen recortada
	msg = new TiXmlElement("nFilasRec");
	msg->LinkEndChild( new TiXmlText( "437" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el número de columnas de la imagen recortada
	msg = new TiXmlElement("nColsRec");
	msg->LinkEndChild( new TiXmlText( "583" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el humbral usado en la máscara de bordes del depth
	msg = new TiXmlElement("thDepthB");
	msg->LinkEndChild( new TiXmlText( "81" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el humbral usado en la máscara de bordes refinada por el color
	msg = new TiXmlElement("thColorB");
	msg->LinkEndChild( new TiXmlText( "7" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el tamaño usado para dilatar la máscara de bordes final
	msg = new TiXmlElement("tamDilate");
	msg->LinkEndChild( new TiXmlText( "5" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el índice del frame inicial
	msg = new TiXmlElement("first_frame");
	msg->LinkEndChild( new TiXmlText( "800" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el índice del frame final
	msg = new TiXmlElement("last_frame");
	msg->LinkEndChild( new TiXmlText( "1100" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de entero, que es el número de modos usados en el MoG
	msg = new TiXmlElement("k_modes");
	msg->LinkEndChild( new TiXmlText( "4" ));
	volume->LinkEndChild( msg );

	//terminamos el volumen entero
	root->LinkEndChild( volume );

	//creamos otro volume de tipo string
	volume = new TiXmlElement("string");

	//creamos un hijo de string, que es el prefijo del directorio tenemos y guardamos
	msg = new TiXmlElement("path");
	msg->LinkEndChild( new TiXmlText( "C:/Users/abb/Documents/imagenes/Pruebas/" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de string, que es el sufijo del directorio del depth siendo .png
	msg = new TiXmlElement("sufPNG");
	msg->LinkEndChild( new TiXmlText( ".png" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de string, que es el sufijo del directorio del color siendo .Jpeg
	msg = new TiXmlElement("sufJPEG");
	msg->LinkEndChild( new TiXmlText( ".Jpeg" ));
	volume->LinkEndChild( msg );

	//terminamos el volumen string
	root->LinkEndChild( volume );

	//creamos otro volume de tipo qstring
	volume = new TiXmlElement("qstring");

	//creamos un hijo de qstring, que es el directorio donte tenemos la imagenes capatadas por el Kinect
	msg = new TiXmlElement("input_dir");
	msg->LinkEndChild( new TiXmlText( "C:/Users/abb/Documents/Etapa2/Secuences/genSeq" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de qstring, que es el directorio donte guardaremos la detección para el Color
	msg = new TiXmlElement("output_dir_color");
	msg->LinkEndChild( new TiXmlText( "C:/Users/abb/Documents/Etapa2/Secuences/genSeqImg5" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de qstring, que es el directorio donte guardaremos la detección para el Depth
	msg = new TiXmlElement("output_dir_depth");
	msg->LinkEndChild( new TiXmlText( "C:/Users/abb/Documents/Etapa2/Secuences/genSeqDepth5" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de qstring, que es el prefijo para los archivos de color
	msg = new TiXmlElement("pref_color");
	msg->LinkEndChild( new TiXmlText( "img_" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de qstring, que es el prefijo para los archivos de depth
	msg = new TiXmlElement("pref_depth");
	msg->LinkEndChild( new TiXmlText( "depth_" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de qstring, que es el sufijo para los archivos de color
	msg = new TiXmlElement("suf_color");
	msg->LinkEndChild( new TiXmlText( ".Jpeg" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de qstring, que es el sufijo para los archivos de depth
	msg = new TiXmlElement("suf_depth");
	msg->LinkEndChild( new TiXmlText( ".png" ));
	volume->LinkEndChild( msg );

	//creamos un hijo de qstring, que es el sufijo para los archivos de detección del FG
	msg = new TiXmlElement("suf_detect");
	msg->LinkEndChild( new TiXmlText( ".tif" ));
	volume->LinkEndChild( msg );

	//terminamos el volumen qstring
	root->LinkEndChild( volume );

	//terminamos cualquier hijo de la clase doc
	doc.LinkEndChild( root );

	//guardamos el archivo con el nombre especificado
	doc.SaveFile(pFilename);

}
