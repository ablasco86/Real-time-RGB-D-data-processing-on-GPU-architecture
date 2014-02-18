#include "AuxFuncs.h"
//#include "math.h"

// Allocates a matrix with Gaussian float entries.
void filterInit(float* data, int size, float sigmas)
{
    for (int i = 0; i < size; ++i){
		for (int j = 0; j < size; ++j){
			int x2= (i-size/2)*(i-size/2)+(j-size/2)*(j-size/2);
			//float x= sqrt(aux);
			float A= (x2)/(2*sigmas*sigmas);
			data[i*size + j]= exp(-A);//(2*PI*sigmas*sigmas);
			//data[i*size + j]= exp(-A);
		}
	} 
}

void filterInit1D(float* data, int size, float sigmas)
{
    for (int i = 0; i < size; ++i){
		int x2= (i-size/2)*(i-size/2);
		float A= (x2)/(2*sigmas*sigmas);
		data[i]= exp(-A);//(2*PI*sigmas*sigmas);
	} 
}

// Allocates a sobel operator float entries.
void createSobelH(int* data)
{
    data[0] = 1; data[1] = 0; data[2] = -1;
	data[3] = 2; data[4] = 0; data[5] = -2;
	data[6] = 1; data[7] = 0; data[8] = -1;
}

void createSobelV(int* data)
{
    data[0] = 1;   data[1] = 2;  data[2] = 1;
	data[3] = 0;   data[4] = 0;	 data[5] = 0;
	data[6] = -1;  data[7] = -2; data[8] = -1;	             
}

void pre_calculation(float* data, int size, float sigmar){
	for (int i = 0; i < size; ++i){
		float A = i*i / (2*sigmar*sigmar);
		data[i] = exp(-A);
	}
}

void pre_sigmaR(float* data, int size, float a, float b, float c){
	for (int i = 0; i < size; ++i){
		data[i] = c + b*3*i + a*3*i*3*i;
	}
}

/*----------------------------------------------------------*
* La función Conversion pasa el fichero .txt en el cual esta*
* el mapa de profundidad a una matriz(Mat).                 *
*                                                           *
* Las entradas es el fichero .txt de la información del     *
* depth y su tamaño.                                        *
* La salida es la matriz con la información del depth.      *
*--------------------------------------—--------------------*/
void conversion(const char* fileName, int &filas, int &columnas, cv::Mat &imagen)
{

	FILE * pFile;
	pFile = fopen (fileName,"r");
	rewind (pFile);

	for(int r=0;r<filas;r++) {

		short int* ptrimg = imagen.ptr<short int>(r);

		for(int c=0;c<columnas;c++){

			fscanf(pFile,"%d",&ptrimg[c]);

		}

	}

}

/*!
Write in a file the data container in the vector #tsVector

@param str name of the file destination.
@param vector containing timestamp information.
*/

void writeVector(const char * str, vector<float> &dataVector){

	FILE *pFile;
	pFile = fopen(str,"w");
	for(int i=0; i< (int)dataVector.size(); i++ )
		fprintf(pFile,"%f\n",dataVector[i]);
	fclose(pFile);


}
