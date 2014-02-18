#include "BilateralFilter.h"
#include "generalcuda.h"
#include <texture_types.h>
#include <texture_fetch_functions.h>

__constant__ float mask[FILTER_SIZE * FILTER_SIZE];
__constant__ float mask1D[FILTER_SIZE];
__constant__ float pre_exp[256];
__constant__ float pre_sigmar[2048];

texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> CmapTexture;


extern "C" void setDistanceMask(float *h_Kernel, size_t mem_size){
    CUDA_SAFE_CALL (cudaMemcpyToSymbol(mask, h_Kernel, mem_size));
}

extern "C" void setDistanceMask1D(float *h_Kernel, size_t mem_size){
    CUDA_SAFE_CALL (cudaMemcpyToSymbol(mask1D, h_Kernel, mem_size));
}

extern "C" void setPre_calculation(float *h_Kernel, size_t mem_size){
    CUDA_SAFE_CALL (cudaMemcpyToSymbol(pre_exp, h_Kernel, mem_size));
}

extern "C" void setPre_sigmaR(float *h_Kernel, size_t mem_size){
    CUDA_SAFE_CALL (cudaMemcpyToSymbol(pre_sigmar, h_Kernel, mem_size));
}

extern "C" void setTextureRellenado(DetectedParameters const *parameters, cudaArray **CmapArray)
{
    CmapTexture.addressMode[0] = cudaAddressModeBorder;
    CmapTexture.addressMode[1] = cudaAddressModeBorder;
    CmapTexture.filterMode = cudaFilterModePoint;
    CmapTexture.normalized = false;
    CUDA_SAFE_CALL (cudaBindTextureToArray(CmapTexture, *CmapArray));
}

// Realiza el filtrado bilateral conjunto sobre una imagen de depth fuera de la zona de incertidumbre
__global__ void imageBilateralFilterKernel(const unsigned short int* input, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, float a, float b, float c)
{
	float sum= 0.0f;
	float norm= 0.0f;
	float weightR;
	int dif;
	float sigmar;
	int _2sigmar2;
	int filterCenter = FILTER_SIZE/2;

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(y >= nl || x >= nc)
		return;

	unsigned short int pixelD = input[y*nc + x];
	//unsigned short int pixelD = modelo[y*nc + x];
	/*if (maskDNC[y*nc + x] == 255 && modelo[y*nc + x] > input[y*nc + x]){
		output[y*nc + x] = input[y*nc + x];
		return;
	}*/

	if (pixelD == 0){
		output[y*nc + x] = pixelD;
		return;
	}

	if (maskB[y*nc + x] == 0 && foreground[y*nc + x] == 0 && maskDNC[y*nc + x] == 0){
		// calcular varianza en funcion de la profundidad del pixel (ruido dependiente con la distancia)
		//sigmar = c + b*input[y*nc + x] + a*input[y*nc + x]*input[y*nc + x];
		//sigmar = (c + b*pixelD + a*pixelD*pixelD)*100000000;
		sigmar = pre_sigmar[pixelD/3]*100000000;
		_2sigmar2 = 2*sigmar*sigmar;

		for (int i = 0; i < FILTER_SIZE; i++){
			if ((y-filterCenter+i >= 0) && (y-filterCenter+i < nl)){
				for (int j = 0; j < FILTER_SIZE; j++){
					if ((x-filterCenter+j >= 0) && (x-filterCenter+j < nc) && input[(y-filterCenter+i)*nc + (x-filterCenter+j)] > 0 && foreground[(y-filterCenter+i)*nc + (x-filterCenter+j)] == 0 && maskB[(y-filterCenter+i)*nc + (x-filterCenter+j)] == 0){ //&& maskDNC[(y-F.height/2+i)*nc + (x-(F.width/2-j))] == 0){
						if (maskDNC[(y-filterCenter+i)*nc + (x-filterCenter+j)] == 255 /*&& modelo[(y-filterCenter+i)*nc + (x-filterCenter+j)] > input[(y-filterCenter+i)*nc + (x-filterCenter+j)]*/)
							continue;
						//dif = modelo[(y-F.height/2+i)*nc + (x-(F.width/2-j))] - input[y*nc + x]; 
						dif = modelo[(y-filterCenter+i)*nc + (x-filterCenter+j)] - pixelD; 
						float A = 100000000*100000000*dif*dif /_2sigmar2;
						A=-A;
						//float A= ((dif*dif)>>1) /(sigmar*sigmar);
						//weightR = exp(-A)/(2*PI*sigmar*sigmar);
						weightR = __expf(A);//(2*PI*sigmar*sigmar);
						//weightR = 1+A/*+(A*A/2)*//*+(A*A*A/6)*//*+(A*A*A*A/24)*//*+(A*A*A*A*A/120)*//*+(A*A*A*A*A*A/720)*/;
						//weightR = __expf(-((dif*dif) /(2*sigmar*sigmar)));//(2*PI*sigmar*sigmar);
						norm += mask[i*FILTER_SIZE+j] * weightR;
						//norm += __fmul_[rn,rz,ru,rd](mask[i*FILTER_SIZE+j], weightR);
						sum += input[(y-filterCenter+i)*nc + (x-filterCenter+j)] * mask[i*FILTER_SIZE+j] * weightR;
					}
				}
			}
		}
		output[y*nc + x] = sum/norm;
	}
	else if (foreground[y*nc + x] == 255){
		// calcular varianza en funcion de la profundidad del pixel (ruido dependiente con la distancia)
		//sigmar = c + b*input[y*nc + x] + a*input[y*nc + x]*input[y*nc + x];
		//sigmar = (c + b*pixelD + a*pixelD*pixelD)*100000000;
		sigmar = pre_sigmar[pixelD/3]*100000000;
		_2sigmar2 = 2*sigmar*sigmar;

		for (int i = 0; i < FILTER_SIZE; i++){
			if ((y-filterCenter+i >= 0) && (y-filterCenter+i < nl)){
				for (int j = 0; j < FILTER_SIZE; j++){
					if ((x-filterCenter+j >= 0) && (x-filterCenter+j < nc) && input[(y-filterCenter+i)*nc + (x-filterCenter+j)] > 0 && foreground[(y-filterCenter+i)*nc + (x-filterCenter+j)] == 255 /*&& maskB[(y-filterCenter+i)*nc + (x-filterCenter+j)] == 0*/){
						//dif = input[(y-F.height/2+i)*nc + (x-(F.width/2-j))] - input[y*nc + x]; 
						dif = input[(y-filterCenter+i)*nc + (x-filterCenter+j)] - pixelD; 
						float A= 100000000*100000000*(dif*dif) /_2sigmar2;
						A=-A;
						//weightR = exp(-A)/(2*PI*sigmar*sigmar);
						weightR = __expf(A);//(2*PI*sigmar*sigmar);
						//weightR = 1+A/*+(A*A/2)*//*+(A*A*A/6)*//*+(A*A*A*A/24)*//*+(A*A*A*A*A/120)*//*+(A*A*A*A*A*A/720)*/;
						norm += mask[i*FILTER_SIZE+j] * weightR;
						sum += input[(y-filterCenter+i)*nc + (x-filterCenter+j)] * mask[i*FILTER_SIZE+j] * weightR;
					}
				}
			}
		}
		output[y*nc + x] = sum/norm;
	}
	else
		output[y*nc + x] = pixelD;
}

extern "C" void imageBilateralFilter(const unsigned short int* input, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, float a, float b, float c)
{
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.y-1) / threads.y);

	// Invoke kernel
	imageBilateralFilterKernel<<< grid, threads >>>(input, modelo, maskB, maskDNC, foreground, output, nl, nc, a, b, c);
}

// Realiza el filtrado bilateral conjunto sobre una imagen de depth fuera de la zona de incertidumbre (aproximación filtro separable horizontal)
__global__ void imageBilateralFilterKernelH(const unsigned short int* input, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, float a, float b, float c)
{
	float sum= 0.0f;
	float norm= 0.0f;
	float weightR;
	int dif;
	float sigmar;
	int _2sigmar2;
	int filterCenter = FILTER_SIZE/2;

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(y >= nl || x >= nc)
		return;

	unsigned short int pixelD = input[y*nc + x];
	/*if (maskDNC[y*nc + x] == 255 && modelo[y*nc + x] > input[y*nc + x]){
		output[y*nc + x] = input[y*nc + x];
		return;
	}*/

	if (pixelD == 0){
		output[y*nc + x] = pixelD;
		return;
	}

	if (maskB[y*nc + x] == 0 && foreground[y*nc + x] == 0 && maskDNC[y*nc + x] == 0){
		// calcular varianza en funcion de la profundidad del pixel (ruido dependiente con la distancia)
		//sigmar = c + b*input[y*nc + x] + a*input[y*nc + x]*input[y*nc + x];
		sigmar = (c + b*pixelD + a*pixelD*pixelD)*100000000;
		_2sigmar2 = 2*sigmar*sigmar;

		//for (int i = 0; i < FILTER_SIZE; i++){
			//if ((y-filterCenter+i >= 0) && (y-filterCenter+i < nl)){
				for (int j = 0; j < FILTER_SIZE; j++){
					if ((x-filterCenter+j >= 0) && (x-filterCenter+j < nc) && input[y*nc + (x-filterCenter+j)] > 0 && foreground[y*nc + (x-filterCenter+j)] == 0 && maskB[y*nc + (x-filterCenter+j)] == 0){ //&& maskDNC[(y-F.height/2+i)*nc + (x-(F.width/2-j))] == 0){
						if (maskDNC[y*nc + (x-filterCenter+j)] == 255 /*&& modelo[y*nc + (x-filterCenter+j)] > input[y*nc + (x-filterCenter+j)]*/)
							continue;
						//dif = modelo[(y-F.height/2+i)*nc + (x-(F.width/2-j))] - input[y*nc + x]; 
						dif = modelo[y*nc + (x-filterCenter+j)] - pixelD; 
						float A = 100000000*100000000*dif*dif /_2sigmar2;
						//float A= ((dif*dif)>>1) /(sigmar*sigmar);
						//weightR = exp(-A)/(2*PI*sigmar*sigmar);
						weightR = __expf(-A);//(2*PI*sigmar*sigmar);
						//weightR = __expf(-((dif*dif) /(2*sigmar*sigmar)));//(2*PI*sigmar*sigmar);
						norm += mask1D[j] * weightR;
						//norm += __fmul_[rn,rz,ru,rd](mask[i*FILTER_SIZE+j], weightR);
						sum += input[y*nc + (x-filterCenter+j)] * mask1D[j] * weightR;
					}
				}
			//}
		//}
		output[y*nc + x] = sum/norm;
	}
	else if (foreground[y*nc + x] == 255){
		// calcular varianza en funcion de la profundidad del pixel (ruido dependiente con la distancia)
		//sigmar = c + b*input[y*nc + x] + a*input[y*nc + x]*input[y*nc + x];
		sigmar = (c + b*pixelD + a*pixelD*pixelD)*100000000;
		_2sigmar2 = 2*sigmar*sigmar;

		//for (int i = 0; i < FILTER_SIZE; i++){
			//if ((y-filterCenter+i >= 0) && (y-filterCenter+i < nl)){
				for (int j = 0; j < FILTER_SIZE; j++){
					if ((x-filterCenter+j >= 0) && (x-filterCenter+j < nc) && input[y*nc + (x-filterCenter+j)] > 0 && foreground[y*nc + (x-filterCenter+j)] == 255/* && maskB[(y-F.height/2+i)*nc + (x-(F.width/2-j))] == 0*/){
						//if (maskDNC[y*nc + (x-filterCenter+j)] == 255 /*&& modelo[y*nc + (x-filterCenter+j)] > input[y*nc + (x-filterCenter+j)]*/)
						//	continue;
						//dif = input[(y-F.height/2+i)*nc + (x-(F.width/2-j))] - input[y*nc + x]; 
						dif = input[y*nc + (x-filterCenter+j)] - pixelD; 
						float A= 100000000*100000000*(dif*dif) /_2sigmar2;
						//weightR = exp(-A)/(2*PI*sigmar*sigmar);
						weightR = __expf(-A);//(2*PI*sigmar*sigmar);
						norm += mask1D[j] * weightR;
						sum += input[y*nc + (x-filterCenter+j)] * mask1D[j] * weightR;
					}
				}
			//}
		//}
		output[y*nc + x] = sum/norm;
	}
	else
		output[y*nc + x] = pixelD;
}

extern "C" void imageBilateralFilterH(const unsigned short int* input, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, float a, float b, float c)
{
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.y-1) / threads.y);

	// Invoke kernel
	imageBilateralFilterKernelH<<< grid, threads >>>(input, modelo, maskB, maskDNC, foreground, output, nl, nc, a, b, c);
}

// Realiza el filtrado bilateral conjunto sobre una imagen de depth fuera de la zona de incertidumbre (aproximación filtro separable vertical)
__global__ void imageBilateralFilterKernelV(const unsigned short int* input, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, float a, float b, float c)
{
	float sum= 0.0f;
	float norm= 0.0f;
	float weightR;
	int dif;
	float sigmar;
	int _2sigmar2;
	int filterCenter = FILTER_SIZE/2;

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(y >= nl || x >= nc)
		return;

	unsigned short int pixelD = input[y*nc + x];
	/*if (maskDNC[y*nc + x] == 255 && modelo[y*nc + x] > input[y*nc + x]){
		output[y*nc + x] = input[y*nc + x];
		return;
	}*/

	if (pixelD == 0){
		output[y*nc + x] = pixelD;
		return;
	}

	if (maskB[y*nc + x] == 0 && foreground[y*nc + x] == 0 && maskDNC[y*nc + x] == 0){
		// calcular varianza en funcion de la profundidad del pixel (ruido dependiente con la distancia)
		//sigmar = c + b*input[y*nc + x] + a*input[y*nc + x]*input[y*nc + x];
		sigmar = (c + b*pixelD + a*pixelD*pixelD)*100000000;
		_2sigmar2 = 2*sigmar*sigmar;

		for (int i = 0; i < FILTER_SIZE; i++){
			if ((y-filterCenter+i >= 0) && (y-filterCenter+i < nl)){
				//for (int j = 0; j < FILTER_SIZE; j++){
					if (/*(x-filterCenter+j >= 0) && (x-filterCenter+j < nc) && */input[(y-filterCenter+i)*nc + x] > 0 && foreground[(y-filterCenter+i)*nc + x] == 0 && maskB[(y-filterCenter+i)*nc + x] == 0){ //&& maskDNC[(y-F.height/2+i)*nc + (x-(F.width/2-j))] == 0){
						if (maskDNC[(y-filterCenter+i)*nc + x] == 255 /*&& modelo[(y-filterCenter+i)*nc + x] > input[(y-filterCenter+i)*nc + x]*/)
							continue;
						//dif = modelo[(y-F.height/2+i)*nc + (x-(F.width/2-j))] - input[y*nc + x]; 
						dif = modelo[(y-filterCenter+i)*nc + x] - pixelD; 
						float A = 100000000*100000000*dif*dif /_2sigmar2;
						//float A= ((dif*dif)>>1) /(sigmar*sigmar);
						//weightR = exp(-A)/(2*PI*sigmar*sigmar);
						weightR = __expf(-A);//(2*PI*sigmar*sigmar);
						//weightR = __expf(-((dif*dif) /(2*sigmar*sigmar)));//(2*PI*sigmar*sigmar);
						norm += mask1D[i] * weightR;
						//norm += __fmul_[rn,rz,ru,rd](mask[i*FILTER_SIZE+j], weightR);
						sum += input[(y-filterCenter+i)*nc + x] * mask1D[i] * weightR;
					}
				//}
			}
		}
		output[y*nc + x] = sum/norm;
	}
	else if (foreground[y*nc + x] == 255){
		// calcular varianza en funcion de la profundidad del pixel (ruido dependiente con la distancia)
		//sigmar = c + b*input[y*nc + x] + a*input[y*nc + x]*input[y*nc + x];
		sigmar = (c + b*pixelD + a*pixelD*pixelD)*100000000;
		_2sigmar2 = 2*sigmar*sigmar;

		for (int i = 0; i < FILTER_SIZE; i++){
			if ((y-filterCenter+i >= 0) && (y-filterCenter+i < nl)){
				//for (int j = 0; j < FILTER_SIZE; j++){
					if (/*(x-filterCenter+j >= 0) && (x-filterCenter+j < nc) && */input[(y-filterCenter+i)*nc + x] > 0 && foreground[(y-filterCenter+i)*nc + x] == 255/* && maskB[(y-F.height/2+i)*nc + (x-(F.width/2-j))] == 0*/){
						//if (maskDNC[(y-filterCenter+i)*nc + x] == 255 /*&& modelo[(y-filterCenter+i)*nc + x] > input[(y-filterCenter+i)*nc + x]*/)
						//	continue;
						//dif = input[(y-F.height/2+i)*nc + (x-(F.width/2-j))] - input[y*nc + x]; 
						dif = input[(y-filterCenter+i)*nc + x] - pixelD; 
						float A= 100000000*100000000*(dif*dif) /_2sigmar2;
						//weightR = exp(-A)/(2*PI*sigmar*sigmar);
						weightR = __expf(-A);//(2*PI*sigmar*sigmar);
						norm += mask1D[i] * weightR;
						sum += input[(y-filterCenter+i)*nc + x] * mask1D[i] * weightR;
					}
				//}
			}
		}
		output[y*nc + x] = sum/norm;
	}
	else
		output[y*nc + x] = pixelD;
}

extern "C" void imageBilateralFilterV(const unsigned short int* input, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, float a, float b, float c)
{
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.y-1) / threads.y);

	// Invoke kernel
	imageBilateralFilterKernelV<<< grid, threads >>>(input, modelo, maskB, maskDNC, foreground, output, nl, nc, a, b, c);
}


// Realiza un filtrado bilateral conjunto para rellenar aquellos pixeles sin información que tienen un número mínimo de 
// vecinos fiables
__global__ void imageRellenarKernel(const Matrix F, const unsigned short int* input, const uchar* color, const uchar* modeloColor, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int sigmar, float porcen, int cMin)
{
	float sum= 0.0f;
	float norm= 0.0f;
	float weightR;
	//int dif1, dif2, dif3;
	int3 dif;
	int cont = 0;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(y >= nl || x >= nc)
		return;

	unsigned short int pixelD = input[y*nc + x];

	if (pixelD > 0){
		output[y*nc + x] = pixelD;
		return;
	}

	uchar3 pixelC = {color[y*nc*nch + x*nch], color[y*nc*nch + x*nch + 1], color[y*nc*nch + x*nch + 2]};
	uchar3 pixelM = {modeloColor[y*nc*nch + x*nch], modeloColor[y*nc*nch + x*nch + 1], modeloColor[y*nc*nch + x*nch + 2]};

	//int _2sigmar2 = 200*sigmar*sigmar;
	int minPixel = 10 * porcen * F.height * F.width;

	if (/*input[y*nc + x] == 0 && */foreground[y*nc + x] == 0){
		for (int i = 0; i < F.height; i++){
			if ((y-F.height/2+i >= 0) && (y-F.height/2+i < nl)){
				for (int j = 0; j < F.width; j++){
					if ((x-(F.width/2-j) >= 0) && (x-(F.height/2-j) < nc) && input[(y-F.height/2+i)*nc + (x-(F.width/2-j))] > 0 && foreground[(y-F.height/2+i)*nc + (x-(F.width/2-j))] == 0){
						unsigned char Cmap = tex2D(CmapTexture, x-(F.width/2-j), y-F.height/2+i);
						if (Cmap < 128+cMin)
							continue;
						dif.x = modeloColor[(y-F.height/2+i)*nc*nch + (x-(F.width/2-j))*nch] - pixelM.x; 
						dif.y = modeloColor[(y-F.height/2+i)*nc*nch + (x-(F.width/2-j))*nch + 1] - pixelM.y;
						dif.z = modeloColor[(y-F.height/2+i)*nc*nch + (x-(F.width/2-j))*nch + 2] - pixelM.z;
						//float A= 100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z) /_2sigmar2;
						//weightR = __expf(-A);//(2*PI*sigmar*sigmar);
						weightR = pre_exp[dif.x] * pre_exp[dif.y] * pre_exp[dif.z];
						norm += mask[i*F.width+j] * weightR;
						sum += input[(y-F.height/2+i)*nc + (x-(F.width/2-j))] * mask[i*F.width+j] * weightR;
						cont++;
					}
				}
			}
		}
		if (10*cont > minPixel)
			output[y*nc + x] = sum/norm;
		else
			output[y*nc + x] = pixelD;
	}
	else if (/*input[y*nc + x] == 0 && */foreground[y*nc + x] == 255){
		for (int i = 0; i < F.height; i++){
			if ((y-F.height/2+i >= 0) && (y-F.height/2+i < nl)){
				for (int j = 0; j < F.width; j++){
					if ((x-(F.width/2-j) >= 0) && (x-(F.height/2-j) < nc) && input[(y-F.height/2+i)*nc + (x-(F.width/2-j))] > 0 && foreground[(y-F.height/2+i)*nc + (x-(F.width/2-j))] == 255){
						dif.x = color[(y-F.height/2+i)*nc*nch + (x-(F.width/2-j))*nch] - pixelC.x; 
						dif.y = color[(y-F.height/2+i)*nc*nch + (x-(F.width/2-j))*nch + 1] - pixelC.y;
						dif.z = color[(y-F.height/2+i)*nc*nch + (x-(F.width/2-j))*nch + 2] - pixelC.z;
						//float A= 100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z) /_2sigmar2;
						//weightR = __expf(-A);//(2*PI*sigmar*sigmar);
						weightR = pre_exp[dif.x] * pre_exp[dif.y] * pre_exp[dif.z];
						norm += mask[i*F.width+j] * weightR;
						sum += input[(y-F.height/2+i)*nc + (x-(F.width/2-j))] * mask[i*F.width+j] * weightR;
					}
				}
			}
		}
		output[y*nc + x] = sum/norm;
	}
	else
		output[y*nc + x] = pixelD;
}

extern "C" void imageRellenar(const Matrix F, const unsigned short int* input, const uchar* color, const uchar* modeloColor, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int sigmar, float porcen, int cMin)
{
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.y-1) / threads.y);

	// Invoke kernel
	imageRellenarKernel<<< grid, threads >>>(F, input, color, modeloColor, foreground, output, nl, nc, nch, sigmar, porcen, cMin);
}

// Realiza un filtrado bilateral conjunto para rellenar aquellos pixeles sin información que tienen un número mínimo de 
// vecinos fiables (aproximación separable horizontal)
__global__ void imageRellenarKernelH(const Matrix F, const unsigned short int* input, const uchar* color, const uchar* modeloColor, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int sigmar, float porcen, int cMin)
{
	float sum= 0.0f;
	float norm= 0.0f;
	float weightR;
	//int dif1, dif2, dif3;
	int3 dif;
	int cont = 0;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(y >= nl || x >= nc)
		return;

	unsigned short int pixelD = input[y*nc + x];

	if (pixelD > 0){
		output[y*nc + x] = pixelD;
		return;
	}

	uchar3 pixelC = {color[y*nc*nch + x*nch], color[y*nc*nch + x*nch + 1], color[y*nc*nch + x*nch + 2]};
	uchar3 pixelM = {modeloColor[y*nc*nch + x*nch], modeloColor[y*nc*nch + x*nch + 1], modeloColor[y*nc*nch + x*nch + 2]};

	//int _2sigmar2 = 200*sigmar*sigmar;
	int minPixel = 10 * porcen * F.width;

	if (/*input[y*nc + x] == 0 && */foreground[y*nc + x] == 0){
		//for (int i = 0; i < F.height; i++){
			//if ((y-F.height/2+i >= 0) && (y-F.height/2+i < nl)){
				for (int j = 0; j < F.width; j++){
					if ((x-(F.width/2-j) >= 0) && (x-(F.height/2-j) < nc) && input[y*nc + (x-(F.width/2-j))] > 0 && foreground[y*nc + (x-(F.width/2-j))] == 0){
						unsigned char Cmap = tex2D(CmapTexture, x-(F.width/2-j), y);
						if (Cmap < 128+cMin)
							continue;
						dif.x = modeloColor[y*nc*nch + (x-(F.width/2-j))*nch] - pixelM.x; 
						dif.y = modeloColor[y*nc*nch + (x-(F.width/2-j))*nch + 1] - pixelM.y;
						dif.z = modeloColor[y*nc*nch + (x-(F.width/2-j))*nch + 2] - pixelM.z;
						//float A= 100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z) /_2sigmar2;
						//weightR = __expf(-A);//(2*PI*sigmar*sigmar);
						weightR = pre_exp[dif.x] * pre_exp[dif.y] * pre_exp[dif.z];
						norm += mask1D[j] * weightR;
						sum += input[y*nc + (x-(F.width/2-j))] * mask1D[j] * weightR;
						cont++;
					}
				}
			//}
		//}
		if (10*cont > minPixel)
			output[y*nc + x] = sum/norm;
		else
			output[y*nc + x] = pixelD;
	}
	else {//if (/*input[y*nc + x] == 0 && */foreground[y*nc + x] == 255){
		//for (int i = 0; i < F.height; i++){
			//if ((y-F.height/2+i >= 0) && (y-F.height/2+i < nl)){
				for (int j = 0; j < F.width; j++){
					if ((x-(F.width/2-j) >= 0) && (x-(F.height/2-j) < nc) && input[y*nc + (x-(F.width/2-j))] > 0 && foreground[y*nc + (x-(F.width/2-j))] == 255){
						dif.x = color[y*nc*nch + (x-(F.width/2-j))*nch] - pixelC.x; 
						dif.y = color[y*nc*nch + (x-(F.width/2-j))*nch + 1] - pixelC.y;
						dif.z = color[y*nc*nch + (x-(F.width/2-j))*nch + 2] - pixelC.z;
						//float A= 100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z) /_2sigmar2;
						//weightR = __expf(-A);//(2*PI*sigmar*sigmar);
						weightR = pre_exp[dif.x] * pre_exp[dif.y] * pre_exp[dif.z];
						norm += mask1D[j] * weightR;
						sum += input[y*nc + (x-(F.width/2-j))] * mask1D[j] * weightR;
					}
				}
			//}
		//}
		output[y*nc + x] = sum/norm;
	}
	//else
	//	output[y*nc + x] = pixelD;
}

extern "C" void imageRellenarH(const Matrix F, const unsigned short int* input, const uchar* color, const uchar* modeloColor, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int sigmar, float porcen, int cMin)
{
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.y-1) / threads.y);

	// Invoke kernel
	imageRellenarKernelH<<< grid, threads >>>(F, input, color, modeloColor, foreground, output, nl, nc, nch, sigmar, porcen, cMin);
}

// Realiza un filtrado bilateral conjunto para rellenar aquellos pixeles sin información que tienen un número mínimo de 
// vecinos fiables (aproximación separable vertical)
__global__ void imageRellenarKernelV(const Matrix F, const unsigned short int* input, const uchar* color, const uchar* modeloColor, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int sigmar, float porcen, int cMin)
{
	float sum= 0.0f;
	float norm= 0.0f;
	float weightR;
	//int dif1, dif2, dif3;
	int3 dif;
	int cont = 0;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(y >= nl || x >= nc)
		return;

	unsigned short int pixelD = input[y*nc + x];

	if (pixelD > 0){
		output[y*nc + x] = pixelD;
		return;
	}

	uchar3 pixelC = {color[y*nc*nch + x*nch], color[y*nc*nch + x*nch + 1], color[y*nc*nch + x*nch + 2]};
	uchar3 pixelM = {modeloColor[y*nc*nch + x*nch], modeloColor[y*nc*nch + x*nch + 1], modeloColor[y*nc*nch + x*nch + 2]};

	//int _2sigmar2 = 200*sigmar*sigmar;
	int minPixel = 10 * porcen * F.height;

	if (/*input[y*nc + x] == 0 && */foreground[y*nc + x] == 0){
		for (int i = 0; i < F.height; i++){
			if ((y-F.height/2+i >= 0) && (y-F.height/2+i < nl)){
				//for (int j = 0; j < F.width; j++){
					if (/*(x-(F.width/2-j) >= 0) && (x-(F.height/2-j) < nc) &&*/ input[(y-F.height/2+i)*nc + x] > 0 && foreground[(y-F.height/2+i)*nc + x] == 0){
						unsigned char Cmap = tex2D(CmapTexture, x, y-F.height/2+i);
						if (Cmap < 128+cMin)
							continue;
						dif.x = modeloColor[(y-F.height/2+i)*nc*nch + x*nch] - pixelM.x; 
						dif.y = modeloColor[(y-F.height/2+i)*nc*nch + x*nch + 1] - pixelM.y;
						dif.z = modeloColor[(y-F.height/2+i)*nc*nch + x*nch + 2] - pixelM.z;
						//float A= 100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z) /_2sigmar2;
						//weightR = __expf(-A);//(2*PI*sigmar*sigmar);
						weightR = pre_exp[dif.x] * pre_exp[dif.y] * pre_exp[dif.z];
						norm += mask1D[i] * weightR;
						sum += input[(y-F.height/2+i)*nc + x] * mask1D[i] * weightR;
						cont++;
					}
				//}
			}
		}
		if (10*cont > minPixel)
			output[y*nc + x] = sum/norm;
		else
			output[y*nc + x] = pixelD;
	}
	else{ //if (/*input[y*nc + x] == 0 && */foreground[y*nc + x] == 255){
		for (int i = 0; i < F.height; i++){
			if ((y-F.height/2+i >= 0) && (y-F.height/2+i < nl)){
				//for (int j = 0; j < F.width; j++){
					if (/*(x-(F.width/2-j) >= 0) && (x-(F.height/2-j) < nc) && */input[(y-F.height/2+i)*nc + x] > 0 && foreground[(y-F.height/2+i)*nc + x] == 255){
						dif.x = color[(y-F.height/2+i)*nc*nch + x*nch] - pixelC.x; 
						dif.y = color[(y-F.height/2+i)*nc*nch + x*nch + 1] - pixelC.y;
						dif.z = color[(y-F.height/2+i)*nc*nch + x*nch + 2] - pixelC.z;
						//float A= 100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z) /_2sigmar2;
						//weightR = __expf(-A);//(2*PI*sigmar*sigmar);
						weightR = pre_exp[dif.x] * pre_exp[dif.y] * pre_exp[dif.z];
						norm += mask1D[i] * weightR;
						sum += input[(y-F.height/2+i)*nc + x] * mask1D[i] * weightR;
					}
				//}
			}
		}
		output[y*nc + x] = sum/norm;
	}
	//else
	//	output[y*nc + x] = pixelD;
}

extern "C" void imageRellenarV(const Matrix F, const unsigned short int* input, const uchar* color, const uchar* modeloColor, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int sigmar, float porcen, int cMin)
{
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.y-1) / threads.y);

	// Invoke kernel
	imageRellenarKernelV<<< grid, threads >>>(F, input, color, modeloColor, foreground, output, nl, nc, nch, sigmar, porcen, cMin);
}
