#include "MedianFilter.h"

// Realiza un filtrado de mediana sobre los bordes usando sólo aquellos pixeles con similitud en color
//__global__ void imageMedianFilterKernel(const unsigned short int* input, const uchar* color, const unsigned short int* modelo,  const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, float sigmar, float thc)
__global__ void imageMedianFilterKernel(const unsigned short int* input, const uchar* color, const unsigned short int* modelo,  const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc)
{
	__shared__ unsigned short int window[BLOCK_SIZE*BLOCK_SIZE][FILTER_SIZE*FILTER_SIZE];
	//__shared__ unsigned short int window[BLOCK_SIZE*BLOCK_SIZE][65535];

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(y >= nl || x >= nc)
		return;

	int tid = threadIdx.y * blockDim.y + threadIdx.x;

	//float weightR;
	//float dif1, dif2, dif3;
	//int dif1, dif2, dif3;
	int3 dif;
	int FILTER_SIZE2 = FILTER_SIZE*FILTER_SIZE;
	//int alternar = 0;
	int alternar = -1;
	int cont = 0;
	int desp;
	int pos = (FILTER_SIZE*FILTER_SIZE)/2;
	int center = pos;
	int filterCenter = FILTER_SIZE/2;
	int ncXnch = nc*nch;
	//float _2sigmar2 = 2*sigmar*sigmar;
	//int _2sigmar2 = 200*sigmar*sigmar;
	//int thaux = 4;

	unsigned short int pixelD = input[y*nc + x];

	uchar3 pixelC = {color[y*ncXnch + x*nch], color[y*ncXnch + x*nch + 1], color[y*ncXnch + x*nch + 2]};

	window[tid][center] = 0;
	/*for(int i = 0; i < center; i++){
		window[tid][FILTER_SIZE2-1-i] = 65535;
		window[tid][i] = 0;
	}*/
	//window[tid][pos] = 0;

	//syncthreads();

	if (/*input[y*nc + x] > 0 && */maskB[y*nc + x] == 255 && foreground[y*nc + x] == 0){
		for (int i = 0; i < FILTER_SIZE; i++){
			int yVecindad = y-filterCenter+i;
			if((yVecindad >= 0) && (yVecindad < nl)){
				for (int j = 0; j < FILTER_SIZE; j++){
					int xVecindad = x-filterCenter+j;
					if((xVecindad >= 0) && (xVecindad < nc) && input[yVecindad*nc + xVecindad] > 0){ 
						if (maskDNC[yVecindad*nc + xVecindad] == 255)// && modelo[yVecindad*nc + xVecindad] > input[yVecindad*nc + xVecindad])
							//if (modelo[yVecindad*nc + xVecindad] > input[yVecindad*nc + xVecindad])
								continue;
						//dif1 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch] - color[y*nc*nch + x*nch]; 
						//dif1 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch] - pixelB; 
						dif.x = color[yVecindad*ncXnch + xVecindad*nch] - pixelC.x; 
						//dif2 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 1] - color[y*nc*nch + x*nch + 1];
						//dif2 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 1] - pixelG;
						dif.y = color[yVecindad*ncXnch + xVecindad*nch + 1] - pixelC.y;
						//dif3 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 2] - color[y*nc*nch + x*nch + 2];
						//dif3 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 2] - pixelR;
						dif.z = color[yVecindad*ncXnch + xVecindad*nch + 2] - pixelC.z;
						//float A= 100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z)/_2sigmar2;
						//float A =__fdiv_rn (100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z), _2sigmar2);
						//float A= (dif.x*dif.x + dif.y*dif.y + dif.z*dif.z)/468;
						int A = dif.x*dif.x + dif.y*dif.y + dif.z*dif.z;
						//weightR = exp(-A);
						//weightR = __expf(-A);
						//int weightR2 = 10*weightR;
						//if (weightR > thc){
						if (A < thc){
						//if (weightR2 > thaux){
							//window[tid][i*(FILTER_SIZE) + j] = input[(y-FILTER_SIZE/2+i)*nc + (x-(FILTER_SIZE/2-j))];
							desp = cont*alternar;
							alternar *= -1;
							pos += desp;
							window[tid][pos] = input[yVecindad*nc + xVecindad];
							cont++;
						}
					}
				}
			}
		}
		int posNoOrdenar = FILTER_SIZE2 - cont;
		int abajo = (posNoOrdenar+1)/2;
		int arriba = posNoOrdenar/2;

		//syncthreads();

		// Order elements (only half of them)
		//for (int i=0; i<=(FILTER_SIZE*FILTER_SIZE)/2; ++i)
		for (int i=abajo; i<=center; ++i)
		{
			// Find position of minimum element
			int min=i;
			//for (int j=i+1; j<FILTER_SIZE*FILTER_SIZE; ++j)
			for (int j=i+1; j<FILTER_SIZE2-arriba; ++j)
				if (window[tid][j] < window[tid][min])
					min=j;

			// Put found minimum element in its place
			const float temp=window[tid][i];
			window[tid][i]=window[tid][min];
			window[tid][min]=temp;

			//syncthreads();
		}
		//if (window[tid][(FILTER_SIZE*FILTER_SIZE)/2] < 5000)
		output[y*nc + x] = window[tid][center];
		//else
		//	output[y*nc + x] = 0;
	}
	else
		//output[y*nc + x] = input[y*nc + x];
		output[y*nc + x] = pixelD;
}

//extern "C" void imageMedianFilter(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, float sigmar, float thc){
extern "C" void imageMedianFilter(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc){
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.x-1) / threads.y);

	// Invoke kernel
	//imageMedianFilterKernel<<< grid, threads >>>(input, color, modelo, maskB, maskDNC, foreground, output, nl, nc, nch, sigmar, thc);
	imageMedianFilterKernel<<< grid, threads >>>(input, color, modelo, maskB, maskDNC, foreground, output, nl, nc, nch, thc);
}

// Realiza un filtrado de mediana aproximado sobre los bordes usando sólo aquellos pixeles con similitud en color
//__global__ void imageMedianFilterKernel(const unsigned short int* input, const uchar* color, const unsigned short int* modelo,  const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, float sigmar, float thc)
__global__ void imageMedianFilterAproxKernel(const unsigned short int* input, const uchar* color, const unsigned short int* modelo,  const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc)
{
	__shared__ unsigned short int window[BLOCK_SIZE*BLOCK_SIZE][(FILTER_SIZE/2+1)*(FILTER_SIZE/2+1) + (FILTER_SIZE/2)%2];

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(y >= nl || x >= nc)
		return;

	int tid = threadIdx.y * blockDim.y + threadIdx.x;

	//float weightR;
	//float dif1, dif2, dif3;
	//int dif1, dif2, dif3;
	int3 dif;
	int FILTER_SIZE2 = (FILTER_SIZE/2+1)*(FILTER_SIZE/2+1) + (FILTER_SIZE/2)%2; // para que siempre sea impar
	//int alternar = 0;
	int alternar = -1;
	int cont = 0;
	int desp;
	int pos = FILTER_SIZE2/2;
	int center = pos;
	int filterCenter = FILTER_SIZE/2;
	int ncXnch = nc*nch;
	//float _2sigmar2 = 2*sigmar*sigmar;
	//int _2sigmar2 = 200*sigmar*sigmar;
	//int thaux = 4;
	//if (FILTER_SIZE2 % 2 == 0)
	//	pos--;

	unsigned short int pixelD = input[y*nc + x];

	uchar3 pixelC = {color[y*ncXnch + x*nch], color[y*ncXnch + x*nch + 1], color[y*ncXnch + x*nch + 2]};

	window[tid][center] = 0;
	/*for(int i = 0; i < center; i++){
		window[tid][FILTER_SIZE2-1-i] = 65535;
		window[tid][i] = 0;
	}*/

	//syncthreads();

	if (/*input[y*nc + x] > 0 && */maskB[y*nc + x] == 255 && foreground[y*nc + x] == 0){
		for (int i = 0; i < FILTER_SIZE; i+=2){
			int yVecindad = y-filterCenter+i;
			if((yVecindad >= 0) && (yVecindad < nl)){
				for (int j = 0; j < FILTER_SIZE; j+=2){
					int xVecindad = x-filterCenter+j;
					if((xVecindad >= 0) && (xVecindad < nc) && input[yVecindad*nc + xVecindad] > 0){ 
						if (maskDNC[yVecindad*nc + xVecindad] == 255)// && modelo[yVecindad*nc + xVecindad] > input[yVecindad*nc + xVecindad])
							//if (modelo[yVecindad*nc + xVecindad] > input[yVecindad*nc + xVecindad])
								continue;
						//dif1 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch] - color[y*nc*nch + x*nch]; 
						//dif1 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch] - pixelB; 
						dif.x = color[yVecindad*ncXnch + xVecindad*nch] - pixelC.x; 
						//dif2 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 1] - color[y*nc*nch + x*nch + 1];
						//dif2 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 1] - pixelG;
						dif.y = color[yVecindad*ncXnch + xVecindad*nch + 1] - pixelC.y;
						//dif3 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 2] - color[y*nc*nch + x*nch + 2];
						//dif3 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 2] - pixelR;
						dif.z = color[yVecindad*ncXnch + xVecindad*nch + 2] - pixelC.z;
						//float A= 100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z)/_2sigmar2;
						//float A =__fdiv_rn (100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z), _2sigmar2);
						//float A= (dif.x*dif.x + dif.y*dif.y + dif.z*dif.z)/468;
						int A = dif.x*dif.x + dif.y*dif.y + dif.z*dif.z;
						//weightR = exp(-A);
						//weightR = __expf(-A);
						//int weightR2 = 10*weightR;
						//if (weightR > thc){
						if (A < thc){
						//if (weightR2 > thaux){
							//window[tid][i*(FILTER_SIZE) + j] = input[(y-FILTER_SIZE/2+i)*nc + (x-(FILTER_SIZE/2-j))];
							desp = cont*alternar;
							alternar *= -1;
							pos += desp;
							window[tid][pos] = input[yVecindad*nc + xVecindad];
							cont++;
						}
					}
				}
			}
		}
		if ((FILTER_SIZE/2)%2 == 1){
			desp = cont*alternar;
			//alternar *= -1;
			pos += desp;
			window[tid][pos] = input[y*nc + x];
			cont++;
		}

		int posNoOrdenar = FILTER_SIZE2 - cont;
		int abajo = (posNoOrdenar+1)/2;
		int arriba = posNoOrdenar/2;

		// esto así para cuando es par el número total de muestras estudiadas
		/*int abajo = posNoOrdenar/2;
		int arriba = (posNoOrdenar+1)/2;

		if (FILTER_SIZE2 % 2 == 0)
			center--;
		else{
			abajo = (posNoOrdenar+1)/2;
			arriba = posNoOrdenar/2;
		}*/
		//syncthreads();

		// Order elements (only half of them)
		//for (int i=0; i<=(FILTER_SIZE*FILTER_SIZE)/2; ++i)
		for (int i=abajo; i<=center; ++i)
		{
			// Find position of minimum element
			int min=i;
			//for (int j=i+1; j<FILTER_SIZE*FILTER_SIZE; ++j)
			for (int j=i+1; j<FILTER_SIZE2-arriba; ++j)
				if (window[tid][j] < window[tid][min])
					min=j;

			// Put found minimum element in its place
			const float temp=window[tid][i];
			window[tid][i]=window[tid][min];
			window[tid][min]=temp;

			//syncthreads();
		}
		//if (window[tid][(FILTER_SIZE*FILTER_SIZE)/2] < 5000)
			output[y*nc + x] = window[tid][center];
		//else
		//	output[y*nc + x] = 0;
	}
	else
		//output[y*nc + x] = input[y*nc + x];
		output[y*nc + x] = pixelD;
}

//extern "C" void imageMedianFilter(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, float sigmar, float thc){
extern "C" void imageMedianFilterAprox(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc){
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.x-1) / threads.y);

	// Invoke kernel
	//imageMedianFilterKernel<<< grid, threads >>>(input, color, modelo, maskB, maskDNC, foreground, output, nl, nc, nch, sigmar, thc);
	imageMedianFilterAproxKernel<<< grid, threads >>>(input, color, modelo, maskB, maskDNC, foreground, output, nl, nc, nch, thc);
}

// Filtro de mediana horizontal 
__global__ void imageMedianFilterKernelH(const unsigned short int* input, const uchar* color, const unsigned short int* modelo,  const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc)
{
	__shared__ unsigned short int window[BLOCK_SIZE*BLOCK_SIZE][FILTER_SIZE];

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(y >= nl || x >= nc)
		return;

	int tid = threadIdx.y * blockDim.y + threadIdx.x;

	//float weightR;
	//float dif1, dif2, dif3;
	//int dif1, dif2, dif3;
	int3 dif;
	//int FILTER_SIZE2 = FILTER_SIZE*FILTER_SIZE;
	//int alternar = 0;
	int alternar = -1;
	int cont = 0;
	int desp;
	int pos = FILTER_SIZE/2;
	int center = pos;
	int filterCenter = FILTER_SIZE/2;
	int ncXnch = nc*nch;
	//float _2sigmar2 = 2*sigmar*sigmar;
	//int _2sigmar2 = 200*sigmar*sigmar;
	//int thaux = 4;

	unsigned short int pixelD = input[y*nc + x];

	uchar3 pixelC = {color[y*ncXnch + x*nch], color[y*ncXnch + x*nch + 1], color[y*ncXnch + x*nch + 2]};

	window[tid][center] = 0;
	for(int i = 0; i < center; i++){
		window[tid][FILTER_SIZE-1-i] = 65535;
		window[tid][i] = 0;
	}
	//window[tid][pos] = 0;

	//syncthreads();

	if (/*input[y*nc + x] > 0 && */maskB[y*nc + x] == 255 && foreground[y*nc + x] == 0){
		//for (int i = 0; i < FILTER_SIZE; i++){
			//int yVecindad = y-filterCenter+i;
			//if((yVecindad >= 0) && (yVecindad < nl)){
				for (int j = 0; j < FILTER_SIZE; j++){
					int xVecindad = x-filterCenter+j;
					if((xVecindad >= 0) && (xVecindad < nc) && input[y*nc + xVecindad] > 0){ 
						if (maskDNC[y*nc + xVecindad] == 255)// && modelo[yVecindad*nc + xVecindad] > input[yVecindad*nc + xVecindad])
						//	if (modelo[yVecindad*nc + xVecindad] > input[yVecindad*nc + xVecindad])
								continue;
						//dif1 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch] - color[y*nc*nch + x*nch]; 
						//dif1 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch] - pixelB; 
						dif.x = color[y*ncXnch + xVecindad*nch] - pixelC.x; 
						//dif2 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 1] - color[y*nc*nch + x*nch + 1];
						//dif2 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 1] - pixelG;
						dif.y = color[y*ncXnch + xVecindad*nch + 1] - pixelC.y;
						//dif3 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 2] - color[y*nc*nch + x*nch + 2];
						//dif3 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 2] - pixelR;
						dif.z = color[y*ncXnch + xVecindad*nch + 2] - pixelC.z;
						//float A= 100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z)/_2sigmar2;
						//float A =__fdiv_rn (100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z), _2sigmar2);
						//float A= (dif.x*dif.x + dif.y*dif.y + dif.z*dif.z)/468;
						int A = dif.x*dif.x + dif.y*dif.y + dif.z*dif.z;
						//weightR = exp(-A);
						//weightR = __expf(-A);
						//int weightR2 = 10*weightR;
						//if (weightR > thc){
						if (A < thc){
						//if (weightR2 > thaux){
							//window[tid][i*(FILTER_SIZE) + j] = input[(y-FILTER_SIZE/2+i)*nc + (x-(FILTER_SIZE/2-j))];
							desp = cont*alternar;
							alternar *= -1;
							pos += desp;
							window[tid][pos] = input[y*nc + xVecindad];
							cont++;
						}
					}
				}
			//}
		//}
		int posNoOrdenar = FILTER_SIZE - cont;
		int abajo = (posNoOrdenar+1)/2;
		int arriba = posNoOrdenar/2;

		//syncthreads();

		// Order elements (only half of them)
		//for (int i=0; i<=(FILTER_SIZE*FILTER_SIZE)/2; ++i)
		for (int i=abajo; i<=center; ++i)
		{
			// Find position of minimum element
			int min=i;
			//for (int j=i+1; j<FILTER_SIZE*FILTER_SIZE; ++j)
			for (int j=i+1; j<FILTER_SIZE-arriba; ++j)
				if (window[tid][j] < window[tid][min])
					min=j;

			// Put found minimum element in its place
			const float temp=window[tid][i];
			window[tid][i]=window[tid][min];
			window[tid][min]=temp;

			//syncthreads();
		}
		//if (window[tid][(FILTER_SIZE*FILTER_SIZE)/2] < 5000)
			output[y*nc + x] = window[tid][center];
		//else
		//	output[y*nc + x] = 0;
	}
	else
		//output[y*nc + x] = input[y*nc + x];
		output[y*nc + x] = pixelD;
}

//extern "C" void imageMedianFilter(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, float sigmar, float thc){
extern "C" void imageMedianFilterH(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc){
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.x-1) / threads.y);

	// Invoke kernel
	//imageMedianFilterKernel<<< grid, threads >>>(input, color, modelo, maskB, maskDNC, foreground, output, nl, nc, nch, sigmar, thc);
	imageMedianFilterKernelH<<< grid, threads >>>(input, color, modelo, maskB, maskDNC, foreground, output, nl, nc, nch, thc);
}

// Filtrado de mediana vertical
__global__ void imageMedianFilterKernelV(const unsigned short int* input, const uchar* color, const unsigned short int* modelo,  const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc)
{
	__shared__ unsigned short int window[BLOCK_SIZE*BLOCK_SIZE][FILTER_SIZE];

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if(y >= nl || x >= nc)
		return;

	int tid = threadIdx.y * blockDim.y + threadIdx.x;

	//float weightR;
	//float dif1, dif2, dif3;
	//int dif1, dif2, dif3;
	int3 dif;
	//int FILTER_SIZE2 = FILTER_SIZE*FILTER_SIZE;
	//int alternar = 0;
	int alternar = -1;
	int cont = 0;
	int desp;
	int pos = FILTER_SIZE/2;
	int center = pos;
	int filterCenter = FILTER_SIZE/2;
	int ncXnch = nc*nch;
	//float _2sigmar2 = 2*sigmar*sigmar;
	//int _2sigmar2 = 200*sigmar*sigmar;
	//int thaux = 4;

	unsigned short int pixelD = input[y*nc + x];

	uchar3 pixelC = {color[y*ncXnch + x*nch], color[y*ncXnch + x*nch + 1], color[y*ncXnch + x*nch + 2]};

	window[tid][center] = 0;
	for(int i = 0; i < center; i++){
		window[tid][FILTER_SIZE-1-i] = 65535;
		window[tid][i] = 0;
	}
	//window[tid][pos] = 0;

	//syncthreads();

	if (/*input[y*nc + x] > 0 && */maskB[y*nc + x] == 255 && foreground[y*nc + x] == 0){
		for (int i = 0; i < FILTER_SIZE; i++){
			int yVecindad = y-filterCenter+i;
			if((yVecindad >= 0) && (yVecindad < nl)){
				//for (int j = 0; j < FILTER_SIZE; j++){
					//int xVecindad = x-filterCenter+j;
					if(/*(xVecindad >= 0) && (xVecindad < nc) && */input[yVecindad*nc + x] > 0){ 
						if (maskDNC[yVecindad*nc + x] == 255)// && modelo[yVecindad*nc + xVecindad] > input[yVecindad*nc + xVecindad])
						//	if (modelo[yVecindad*nc + xVecindad] > input[yVecindad*nc + xVecindad])
								continue;
						//dif1 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch] - color[y*nc*nch + x*nch]; 
						//dif1 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch] - pixelB; 
						dif.x = color[yVecindad*ncXnch + x*nch] - pixelC.x; 
						//dif2 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 1] - color[y*nc*nch + x*nch + 1];
						//dif2 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 1] - pixelG;
						dif.y = color[yVecindad*ncXnch + x*nch + 1] - pixelC.y;
						//dif3 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 2] - color[y*nc*nch + x*nch + 2];
						//dif3 = color[(y-FILTER_SIZE/2+i)*nc*nch + (x-(FILTER_SIZE/2-j))*nch + 2] - pixelR;
						dif.z = color[yVecindad*ncXnch + x*nch + 2] - pixelC.z;
						//float A= 100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z)/_2sigmar2;
						//float A =__fdiv_rn (100*(dif.x*dif.x + dif.y*dif.y + dif.z*dif.z), _2sigmar2);
						//float A= (dif.x*dif.x + dif.y*dif.y + dif.z*dif.z)/468;
						int A = dif.x*dif.x + dif.y*dif.y + dif.z*dif.z;
						//weightR = exp(-A);
						//weightR = __expf(-A);
						//int weightR2 = 10*weightR;
						//if (weightR > thc){
						if (A < thc){
						//if (weightR2 > thaux){
							//window[tid][i*(FILTER_SIZE) + j] = input[(y-FILTER_SIZE/2+i)*nc + (x-(FILTER_SIZE/2-j))];
							desp = cont*alternar;
							alternar *= -1;
							pos += desp;
							window[tid][pos] = input[yVecindad*nc + x];
							cont++;
						}
					}
				//}
			}
		}
		int posNoOrdenar = FILTER_SIZE - cont;
		int abajo = (posNoOrdenar+1)/2;
		int arriba = posNoOrdenar/2;

		//syncthreads();

		// Order elements (only half of them)
		//for (int i=0; i<=(FILTER_SIZE*FILTER_SIZE)/2; ++i)
		for (int i=abajo; i<=center; ++i)
		{
			// Find position of minimum element
			int min=i;
			//for (int j=i+1; j<FILTER_SIZE*FILTER_SIZE; ++j)
			for (int j=i+1; j<FILTER_SIZE-arriba; ++j)
				if (window[tid][j] < window[tid][min])
					min=j;

			// Put found minimum element in its place
			const float temp=window[tid][i];
			window[tid][i]=window[tid][min];
			window[tid][min]=temp;

			//syncthreads();
		}
		//if (window[tid][(FILTER_SIZE*FILTER_SIZE)/2] < 5000)
			output[y*nc + x] = window[tid][center];
		//else
		//	output[y*nc + x] = 0;
	}
	else
		//output[y*nc + x] = input[y*nc + x];
		output[y*nc + x] = pixelD;
}

//extern "C" void imageMedianFilter(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, float sigmar, float thc){
extern "C" void imageMedianFilterV(const unsigned short int* input, const uchar* color, const unsigned short int* modelo, const uchar* maskB, const uchar* maskDNC, const uchar* foreground, unsigned short int* output, int nl, int nc, int nch, int thc){
	// setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((nc+threads.x-1) / threads.x, (nl+threads.x-1) / threads.y);

	// Invoke kernel
	//imageMedianFilterKernel<<< grid, threads >>>(input, color, modelo, maskB, maskDNC, foreground, output, nl, nc, nch, sigmar, thc);
	imageMedianFilterKernelV<<< grid, threads >>>(input, color, modelo, maskB, maskDNC, foreground, output, nl, nc, nch, thc);
}
