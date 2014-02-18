#include "BordersDetect.h"
#include "generalcuda.h"

__constant__ int sobelH[SOBEL_SIZE*SOBEL_SIZE];
__constant__ int sobelV[SOBEL_SIZE*SOBEL_SIZE];

//extern "C" void setSobelMask(float *h_KernelH, float *h_KernelV, size_t mem_size){
extern "C" void setSobelMask(int *h_KernelH, int *h_KernelV, size_t mem_size){
    CUDA_SAFE_CALL (cudaMemcpyToSymbol(sobelH, h_KernelH, mem_size));
	CUDA_SAFE_CALL (cudaMemcpyToSymbol(sobelV, h_KernelV, mem_size));
}

// Matrix filter kernel called by Main()
__global__ void detectarBordesKernel(const Matrix2 H, const Matrix2 V, const unsigned short int* input, uchar* color, uchar* output, int nl, int nc, int nch, int tam, int thd, int thc){
	int sumH= 0;
	int sumV= 0;
	float sumL= 0.0f;
	float sumL2= 0.0f;
	int normH= 0;
	int normV= 0;
	int normL= 0;
	float aux1;
	float aux2;
	float media;

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if ((nc <= x) || (nl <= y)) { return; }

	// Busqueda de bordes en Depth
	//if (y < nl && x < nc){ //&& input[row*nc + col] > 0){
	for (int i = 0; i < H.height; i++){
		if ((y-H.height/2+i >= 0) && (y-H.height/2+i < nl)){
			for (int j = 0; j < H.width; j+=2){
				if ((x-(H.width/2-j) >= 0) && (x-(H.width/2-j) < nc)){ //&& input[(row-H.height/2+i)*nc + (col-(H.width/2-j))] > 0){
					normH += abs(sobelH[i*H.width+j]);
					//normV += abs(sobelV[i*H.width+j]);
					sumH += input[(y-H.height/2+i)*nc + (x-H.width/2+j)] * sobelH[i*H.width+j];
					//sumV += input[(y-V.height/2+i)*nc + (x-V.width/2+j)] * sobelV[i*V.width+j];
				}
			}
		}
	}

	for (int i = 0; i < V.height; i+=2){
		if ((y-V.height/2+i >= 0) && (y-V.height/2+i < nl)){
			for (int j = 0; j < V.width; j++){
				if ((x-(V.width/2-j) >= 0) && (x-(V.width/2-j) < nc)){ //&& input[(row-H.height/2+i)*nc + (col-(H.width/2-j))] > 0){
					normV += abs(sobelV[i*H.width+j]);
					sumV += input[(y-V.height/2+i)*nc + (x-(V.width/2-j))] * sobelV[i*V.width+j];
				}
			}
		}
	}
	int multH = 10*sumH/normH;
	int multV = 10*sumV/normV;
	//aux1 = sqrt(sumH/normH*sumH/normH + sumV/normV*sumV/normV);
	aux1 = __fsqrt_rn(multH*multH + multV*multV);
	if (aux1 > 10*thd) 
		output[y*nc + x] = 255;
	else{
		output[y*nc + x] = 0;
		return;
	}
		
	// Refinado por el color
	for (int i = 0; i < tam; i++){
		if ((y-tam/2+i >= 0) && (y-tam/2+i < nl)){
			for (int j = 0; j < tam; j++){
				if ((x-(tam/2-j) >= 0) && (x-(tam/2-j) < nc)){ //&& input[(row-H.height/2+i)*nc + (col-(H.width/2-j))] > 0){
					normL++;
					sumL += color[(y-tam/2+i)*nc*nch + (x-(tam/2-j))*nch];
				}
			}
		}
	}
	media = sumL/normL;
	for (int i = 0; i < tam; i++){
		if ((y-tam/2+i >= 0) && (y-tam/2+i < nl)){
			for (int j = 0; j < tam; j++){
				if ((x-(tam/2-j) >= 0) && (x-(tam/2-j) < nc)){ //&& input[(row-H.height/2+i)*nc + (col-(H.width/2-j))] > 0){
					sumL2 += (color[(y-tam/2+i)*nc*nch + (x-(tam/2-j))*nch] - media) * (color[(y-tam/2+i)*nc*nch + (x-(tam/2-j))*nch] - media);
				}
			}
		}
	}
	//aux2 = sqrt(sumL2/normL);
	aux2 = __fsqrt_rn(sumL2/normL);
	if (aux2 < thc)
		output[y*nc + x] = 0;
	//}
}

extern "C" void detectarBordes(const Matrix2 H, const Matrix2 V, const unsigned short int* input, uchar* color, uchar* output, int nl, int nc, int nch, int tam, int thd, int thc){
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.y-1) / threads.y);

	// Invoke kernel
	detectarBordesKernel<<< grid, threads >>>(H, V, input, color, output, nl, nc, nch, tam, thd, thc);
}
