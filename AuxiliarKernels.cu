#include "AuxiliarKernels.h"

// Recorta los bordes de la imagen (tanto Depth como color) correspondientes a zonas sin información del Depth, 
__global__ void recortarKernel(const unsigned short int *inputD, const uchar* inputC, unsigned short int* outputD, uchar* outputC, int IF, int IC, int nl, int nc, int nct, int nch){
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if ((nc <= x) || (nl <= y)) { return; }
	
	outputD[y*nc + x] = inputD[(IF+y)*nct + IC+x];
	outputC[y*nc*nch + x*nch] = inputC[(IF+y)*nct*nch + (IC+x)*nch];
	outputC[y*nc*nch + x*nch + 1] = inputC[(IF+y)*nct*nch + (IC+x)*nch + 1];
	outputC[y*nc*nch + x*nch + 2] = inputC[(IF+y)*nct*nch + (IC+x)*nch + 2];
}

extern "C" void recortar(const unsigned short int *inputD, const uchar* inputC, unsigned short int* outputD, uchar* outputC, int IF, int IC, int nl, int nc, int nct, int nch){
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.y-1) / threads.y);

	// Invoke kernel
	recortarKernel<<< grid, threads >>>(inputD, inputC, outputD, outputC, IF, IC, nl, nc, nct, nch);
}

//__global__ void rellenarBeforeFilteringKernel(const unsigned short int *input, const unsigned short int *modelo, uchar* maskForeground, uchar* maskBordersDD, unsigned short int *output, unsigned short int *outputModel, int nl, int nc){
__global__ void rellenarBeforeFilteringKernel(const unsigned short int *input, const unsigned short int *modelo, uchar* maskForeground, unsigned short int *output, unsigned short int *outputModel, int nl, int nc){
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if ((nc <= x) || (nl <= y)) { return; }

	/*if (maskForeground[y*nc + x] == 255){
		output[y*nc + x] = input[y*nc + x];
		outputModel[y*nc + x] = modelo[y*nc + x];
		return;
	}*/

	// to fill the current image with the model
	if (maskForeground[y*nc + x] == 0 && input[y*nc + x] == 0) //|| (maskBordersDD[y*nc + x] == 255 && input[y*nc + x] < modelo[y*nc + x]))
		output[y*nc + x] = modelo[y*nc + x];
	// if is foreground me do not modify
	else
		output[y*nc + x] = input[y*nc + x];

	// to fill de model with the current image
	if (maskForeground[y*nc + x] == 0 && modelo[y*nc + x] == 0)
		outputModel[y*nc + x] = input[y*nc +x];
	// if is foreground me do not modify
	else
		outputModel[y*nc + x] = modelo[y*nc + x];
}

//extern "C" void rellenarBeforeFiltering(const unsigned short int *input, const unsigned short int *modelo, uchar* maskForeground, uchar* maskBordersDD, unsigned short int *output, unsigned short int *outputModel, int nl, int nc){
extern "C" void rellenarBeforeFiltering(const unsigned short int *input, const unsigned short int *modelo, uchar* maskForeground, unsigned short int *output, unsigned short int *outputModel, int nl, int nc){
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.y-1) / threads.y);

	// Invoke kernel
	//rellenarBeforeFilteringKernel<<< grid, threads >>>(input, modelo, maskForeground, maskBordersDD, output, outputModel, nl, nc);
	rellenarBeforeFilteringKernel<<< grid, threads >>>(input, modelo, maskForeground, output, outputModel, nl, nc);

}


__global__ void rellenarBordersDepthNoColorKernel(const unsigned short int *input, const unsigned short int *modelo, uchar* maskBordes, uchar* maskDNC, unsigned short int *output, int nl, int nc){
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if ((nc <= x) || (nl <= y)) { return; }

	// to fill the current image with the model
	if (maskBordes[y*nc + x] == 255 && maskDNC[y*nc + x] == 255 && modelo[y*nc + x] > input[y*nc + x])
		output[y*nc + x] = modelo[y*nc + x];
	else // do not modify
		output[y*nc + x] = input[y*nc + x];
}


extern "C" void rellenarBordersDepthNoColor(const unsigned short int *input, const unsigned short int *modelo, uchar* maskBordes, uchar* maskDNC, unsigned short int *output, int nl, int nc){
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.y-1) / threads.y);

	// Invoke kernel
	//rellenarBeforeFilteringKernel<<< grid, threads >>>(input, modelo, maskForeground, maskBordersDD, output, outputModel, nl, nc);
	rellenarBordersDepthNoColorKernel<<< grid, threads >>>(input, modelo, maskBordes, maskDNC, output, nl, nc);
}
