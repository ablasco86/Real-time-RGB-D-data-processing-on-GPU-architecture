/*
 * mixture.cpp
 *
 *  Created on: Jan 9, 2013
 *      Authors: dbd & abb
 *
 *	PFC Antonio Blasco de Blas
 *	Desarrollo de algoritmos de filtrado de mapas de profundidad sobre procesadores gráficos GPGPU
 */

#include "stdafx.h"
//#include <FreeImagePlus.h>
#include <tinystr.h>
#include <tinyxml.h>

#include <QtCore/QDebug>
#include <cassert>
//#include <cuda_runtime.h>
#include <channel_descriptor.h>

#include "CrearXML.h"
#include "LeerXML.h"
#include "AuxFuncs.h"
#include "ioconfig.h"
#include "mogconfig.h"
#include "myqdir.h"
#include "generalcuda.h"
#include "cudaimages.h"
#include "cudamog.h"
#include "detectedparameters.h"
#include "AuxiliarKernels.h"
#include "BordersDetect.h"
#include "MedianFilter.h"
#include "BilateralFilter.h"

#define MEASURETIME
//#define MEASURETIME2

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main (int argc, char **argv) {
	// Create XML file
	//CreateXMLFile("C:/Users/abb/Documents/Etapa3/version8/prueba1mog.xml");

	// Load XML file
	TiXmlDocument doc("C:/Users/abb/Documents/Etapa3/Version8/prueba1mog.xml");
	//TiXmlDocument doc(argv[1]);
	doc.LoadFile();

	// Load system parameters
	float sigmaS = ReturnFloat(doc, "sigma_s");
	float sigmaCR = ReturnFloat(doc, "sigma_cr");
	//float thFiltroM = ReturnFloat(doc, "th_filtro_m");
	float porcentaje = ReturnFloat(doc, "porcentaje");
	float aa = ReturnFloat (doc, "a");
	float bb = ReturnFloat (doc, "b");
	float cc = ReturnFloat (doc, "c");
	float sigma0Lum = ReturnFloat (doc, "sigma_0_Lum");
	float sigma0Cab = ReturnFloat (doc, "sigma_0_Cab");
	float sigma0Depth = ReturnFloat (doc, "sigma_0_depth");
	float weight0 = ReturnFloat (doc, "weight_0");
	float lambda = ReturnFloat (doc, "lambda");
	float alphaMin = ReturnFloat (doc, "alpha_min");
	float thresholdDetect = ReturnFloat (doc, "threshold_detect");
	float sigmaMinLum = ReturnFloat (doc, "sigma_min_Lum");
	float sigmaMinCab = ReturnFloat (doc, "sigma_min_Cab");
	float sigmaMinDepth = ReturnFloat (doc, "sigma_min_depth");
	float deltaSigmaDepth = ReturnFloat (doc, "delta_sigma_depth");
	float thWeightMin = ReturnFloat(doc, "th_weight_min");
	float thSigmaMax = ReturnFloat(doc, "th_sigma_max");
	int filterSize = ReturnInt(doc, "filter_size");
	//int sigmaR = ReturnInt(doc, "sigma_r");
	int initFilas = ReturnInt(doc, "init_filas");
	int initCols = ReturnInt(doc, "init_columnas");
	int filasRec = ReturnInt(doc, "n_filas_rec");
	int colsRec = ReturnInt(doc, "n_cols_rec");
	int pStepBytes = colsRec;
	int pStepBytesMorph = ReturnInt(doc, "tam_morph_operator");
	Npp32s nStepBytes = colsRec;
	int tamMorph = pStepBytesMorph;
	int thDepthB = ReturnInt(doc, "th_depth_b");
	int thColorB = ReturnInt(doc, "th_color_b");
	int thFiltroM = ReturnInt(doc, "th_filtro_m");
	//int tamDilate = ReturnInt(doc, "tam_dilate");
	//int tamErode = ReturnInt(doc, "tam_erode");
	int firstFrame = ReturnInt(doc, "first_frame");
	int lastFrame = ReturnInt(doc, "last_frame");
	int kModes = ReturnInt(doc, "k_modes");
	int cMin = ReturnInt(doc, "c_min");
	String path = ReturnString (doc, "path");
	String sufDepthS = ReturnString(doc, "suf_depthS");
	String sufColorS = ReturnString(doc, "suf_colorS");
	QString inputDir = ReturnQString (doc, "input_dir");
	QString outputDirColor = ReturnQString (doc, "output_dir_color");
	QString outputDirDepth = ReturnQString (doc, "output_dir_depth");
	QString outputDirEnsamble = ReturnQString (doc, "output_dir_ensamble");
	QString prefColor = ReturnQString (doc, "pref_color");
	QString prefDepth = ReturnQString (doc, "pref_depth");
	QString prefEnsamble = ReturnQString (doc, "pref_ensamble");
	QString sufColor = ReturnQString (doc, "suf_color");	
	QString sufDepth = ReturnQString (doc, "suf_depth");
	QString sufDetect = ReturnQString (doc, "suf_detect");

	float firstFrameTime;
	float totalTime = 0.0f;
	float meanTime;

	char tmpPath [100];
	const IOConfig ioconfigColor(IOConfig().setFilenameTemplate(prefColor + "%d" + sufColor)
                                      .setFirstFrameNumber(firstFrame)
                                      .setLastFrameNumber(lastFrame)
                                      .setInputDir(inputDir)
                                      .setOutputDir(outputDirColor));
    MyQDir outdirColor(ioconfigColor.outputDir.absolutePath());
    if (!outdirColor.exists()) {
        assert(outdirColor.mkpath("."));
    } else {
        assert(outdirColor.empty());
    }
    const IOConfig ioconfigDepth(IOConfig().setFilenameTemplate(prefDepth + "%d" + sufDepth)
                                      .setFirstFrameNumber(firstFrame)
                                      .setLastFrameNumber(lastFrame)
                                      .setInputDir(inputDir)
                                      .setOutputDir(outputDirDepth));
    MyQDir outdirDepth(ioconfigDepth.outputDir.absolutePath());
    if (!outdirDepth.exists()) {
        assert(outdirDepth.mkpath("."));
    } else {
        assert(outdirDepth.empty());
    }
	const IOConfig ioconfigEnsamble(IOConfig().setFilenameTemplate(prefEnsamble + "%d")
                                      .setFirstFrameNumber(firstFrame)
                                      .setLastFrameNumber(lastFrame)
                                      .setInputDir(inputDir)
                                      .setOutputDir(outputDirEnsamble));
    MyQDir outdirEnsamble(ioconfigEnsamble.outputDir.absolutePath());
    if (!outdirEnsamble.exists()) {
        assert(outdirEnsamble.mkpath("."));
    } else {
        assert(outdirEnsamble.empty());
    }
	// Parámetros para el MoG del color
	const MoGConfig mogconfigColor = MoGConfigFactory().set_k(kModes)
                                                  .set_sigma_0 (sigma0Lum)
												  .set_sigmaCab_0 (sigma0Cab)
                                                  .set_w_0 (weight0)
                                                  .set_lambda (lambda)
                                                  .set_alpha_min (alphaMin)
                                                  .set_thresh (thresholdDetect)
                                                  .set_sigma_min (sigmaMinLum)
												  .set_sigmaCab_min (sigmaMinCab)
                                                  .toStruct();
    qDebug() << mogconfigColor.thresh;
	// Parámetros para el MoG del depth
    const MoGConfig mogconfigDepth = MoGConfigFactory().set_k(kModes)
                                                  .set_sigma_0 (sigma0Depth)
                                                  .set_w_0 (weight0)
                                                  .set_lambda (lambda)
                                                  .set_alpha_min (alphaMin)
                                                  .set_thresh (thresholdDetect)
                                                  .set_sigma_min(sigmaMinDepth)
                                                  .toStruct();
    qDebug() << mogconfigDepth.thresh;

	Mat inDepth, inColor;
	//Mat inColor;
	inColor = imread(ioconfigColor.inputDir.absoluteFilePath(QString().sprintf(ioconfigColor.filenameTemplate.toUtf8().data(), ioconfigColor.first)).toStdString().c_str());
	inDepth = imread(ioconfigDepth.inputDir.absoluteFilePath(QString().sprintf(ioconfigDepth.filenameTemplate.toUtf8().data(), ioconfigDepth.first)).toStdString().c_str(), CV_LOAD_IMAGE_ANYDEPTH);
	//Mat inDepth(inColor.size(), CV_16UC1, Scalar(0));
	//char fileDepthTXT [100];
	//sprintf(fileDepthTXT,"%s%d%s", path.c_str(), 0, sufDepthS.c_str());
	//conversion(fileDepthTXT, inColor.rows, inColor.cols, inDepth);


	//Mat inColorLab(inColor.size(), inColor.type(), Scalar(0)); // image Color Lab
	Mat inColorYCrCb(inColor.size(), inColor.type(), Scalar(0)); // image Color Lab
	//Mat inColorRecLab(filasRec, colsRec, CV_8UC3, Scalar(0)); // recorted image for color
	Mat inColorRecYCrCb(filasRec, colsRec, CV_8UC3, Scalar(0)); // recorted image for color
	Mat inDepthRec(filasRec, colsRec, CV_16UC1, Scalar(0)); // recorted image for depth
	Mat inColor32(filasRec, colsRec, CV_8UC4, Scalar(0)); // 32 bits color image
	Mat outColor(filasRec, colsRec, CV_8UC1); // Color foreground detection
	Mat outDepth(filasRec, colsRec, CV_8UC1); // Depth foreground detection
	Mat filledDetection(filasRec, colsRec, CV_8UC1); 
	//Mat elementErode = getStructuringElement(MORPH_ELLIPSE,Size(tamErode,tamErode),Point(-1,-1));
	//Mat elementDilate = getStructuringElement(MORPH_ELLIPSE,Size(tamDilate,tamDilate),Point(-1,-1));
	Mat morphOperator(tamMorph, tamMorph, CV_8UC1, Scalar(255));
	Mat erodedDetection(filasRec, colsRec, CV_8UC1);
	Mat borders(filasRec, colsRec, CV_8UC1);
	Mat bordersColor(filasRec, colsRec, CV_8UC1);
	Mat finalDetection(filasRec, colsRec, CV_8UC1, Scalar(0)); // Final foreground detection
	Mat bordersDepthNoColor(filasRec, colsRec, CV_8UC1, Scalar(0)); // borders detected by depth but not in the final detection
	Mat mascaraBordes(filasRec, colsRec, CV_8UC1, Scalar(0)); // mask image initialize to black
	Mat bordersFiltered(filasRec, colsRec, CV_16UC1, Scalar(0)); // image with the borders filtered (Median Filter)
	Mat bordersFilteredH(filasRec, colsRec, CV_16UC1, Scalar(0)); // image with the borders filtered (Median Filter H)
	Mat filtered(filasRec, colsRec, CV_16UC1, Scalar(0)); // filtered image initialize to black
	Mat filteredH(filasRec, colsRec, CV_16UC1, Scalar(0)); // filtered image initialize to black
	Mat rellena(filasRec, colsRec, CV_16UC1, Scalar(0)); // filled image initialize to black
	Mat modeloDepth(filasRec, colsRec, CV_16UC1, Scalar(0)); // Background stable model for depth
	Mat modeloDF(filasRec, colsRec, CV_16UC1, Scalar(0)); // filled before filtering
	Mat modeloColor(filasRec, colsRec, CV_8UC3, Scalar(0)); // recorted image for color
	Mat rellena2(filasRec, colsRec, CV_16UC1, Scalar(0)); // filled borders depth no color with the model
	Mat processedH(filasRec, colsRec, CV_16UC1, Scalar(0)); // filled image (separable HJBF)
	Mat processed(filasRec, colsRec, CV_16UC1, Scalar(0)); // final image

	//NVIDIA Performace Primitives
	NppiSize oMaskSize = {tamMorph, tamMorph};
	NppiSize oSizeROIRec = {colsRec - oMaskSize.width + 1, filasRec - oMaskSize.height + 1};
	NppiSize oSizeROI = {colsRec, filasRec};
	NppiPoint oAnchor = {0, 0};

	const DetectedParameters parametersColor = DetectedParametersFactory().setHeight(inColorRecYCrCb.rows)
                                                                     .setWidth(inColorRecYCrCb.cols)
                                                                     .toStruct();
	const DetectedParameters parametersDepth = DetectedParametersFactory().setHeight(inDepthRec.rows)
                                                                     .setWidth(inDepthRec.cols)
                                                                     .toStruct();

	int nl= inColor.rows; // number of lines
	int nc= inColor.cols; // number of colums
	int nch= inColor.channels(); // number of channels

	//size_t memSizeDepth = nl*nc*sizeof(unsigned short int); // size of original Depth image
	//size_t memSizeColor = nl*nc*nch*sizeof(uchar); // size of original RGB image
	size_t memSizeDR = filasRec*colsRec*sizeof(unsigned short int); // size of recorted Depth image
	size_t memSizeCR = filasRec*colsRec*nch*sizeof(uchar); // size of recorted Color image
	size_t memSizeMaskB = filasRec*colsRec*sizeof(uchar); // size of edges mask
	size_t memSizeNPPMorph = tamMorph*tamMorph*sizeof(Npp8u); // size of morpholigical operator

	// Host pointers
	//unsigned short int* ptrInDepth_h = inDepth.ptr<unsigned short int>();
	//uchar* ptrInColor_h = inColorLab.ptr<uchar>();
	unsigned short int* ptrInDepthRec_h = inDepthRec.ptr<unsigned short int>();
	//uchar* ptrInColorRecLab_h = inColorRecLab.ptr<uchar>();
	uchar* ptrInColorRecYCrCb_h = inColorRecYCrCb.ptr<uchar>();
	Npp8u* ptrBordersDNC_h = bordersDepthNoColor.ptr<Npp8u>(); // Bordes de la detección en Depth
	Npp8u* ptrMaskB_h = mascaraBordes.ptr<Npp8u>();
	//Npp8u* ptrMorph_h = morphOperator.ptr<Npp8u>();
	unsigned short int* ptrFB_h = bordersFiltered.ptr<unsigned short int>(); 
	unsigned short int* ptrFBH_h = bordersFiltered.ptr<unsigned short int>(); 
	unsigned short int* ptrFiltered_h = filtered.ptr<unsigned short int>();
	unsigned short int* ptrFilteredH_h = filtered.ptr<unsigned short int>();
	unsigned short int* ptrRellena_h = rellena.ptr<unsigned short int>();
	unsigned short int* ptrModeloDepth_h = modeloDepth.ptr<unsigned short int>();
	Npp8u* ptrFinalDetection_h = finalDetection.ptr<Npp8u>();
	unsigned short int* ptrModeloDF_h = modeloDF.ptr<unsigned short int>();
	uchar* ptrModeloColor_h = modeloColor.ptr<uchar>();
	unsigned short int* ptrRellena2_h = rellena2.ptr<unsigned short int>();
	unsigned short int* ptrProcessedH_h = processedH.ptr<unsigned short int>();
	unsigned short int* ptrProcessed_h = processed.ptr<unsigned short int>();
	Npp8u* ptrOutColor_h = outColor.ptr<Npp8u>();
	Npp8u* ptrFilledDetection_h = filledDetection.ptr<Npp8u>();
	Npp8u* ptrErodedDetection_h = erodedDetection.ptr<Npp8u>();
	Npp8u* ptrBorders_h = borders.ptr<Npp8u>();
	Npp8u* ptrBordersColor_h = bordersColor.ptr<Npp8u>();

	// Device pointers
	//unsigned short int* ptrInDepth_d;
	//uchar* ptrInColorLab_d;
	//uchar* ptrInColorYCrCb_d;
	Npp8u* ptrBordersDNC_d;
	unsigned short int* ptrInDepthRec_d;
	//uchar* ptrInColorRecLab_d;
	uchar* ptrInColorRecYCrCb_d;
	//uchar* ptrMaskB_d;
	Npp8u* ptrMaskB_d;
	Npp8u* ptrMorph_d;
	unsigned short int* ptrFB_d; 
	unsigned short int* ptrFBH_d;
	unsigned short int* ptrFiltered_d;
	unsigned short int* ptrFilteredH_d;
	unsigned short int* ptrRellena_d;
	unsigned short int* ptrModeloDepth_d;
	Npp8u* ptrFinalDetection_d;
	unsigned short int* ptrModeloDF_d;
	uchar* ptrModeloColor_d;
	unsigned short int* ptrRellena2_d;
	unsigned short int* ptrProcessedH_d;
	unsigned short int* ptrProcessed_d;
	Npp8u* ptrOutColor_d;
	Npp8u* ptrFilledDetection_d;
	Npp8u* ptrErodedDetection_d;
	Npp8u* ptrBorders_d;
	Npp8u* ptrBordersColor_d;

	/*Npp8u ptrMorph_h[9] =	 {0,1,0,
							  1,1,1,
							  0,1,0};*/

	/*Npp8u ptrMorph_h[25] =	 {0,0,1,0,0,
							  0,1,1,1,0,
							  1,1,1,1,1,
							  0,1,1,1,0,
							  0,0,1,0,0};*/

	/*Npp8u ptrMorph_h[49] =	 {0,0,0,1,0,0,0,
							  0,0,1,1,1,0,0,
							  0,1,1,1,1,1,0,
							  1,1,1,1,1,1,1,
							  0,1,1,1,1,1,0,
							  0,0,1,1,1,0,0,
							  0,0,0,1,0,0,0};*/

	Npp8u ptrMorph_h[81] =	 {0,0,0,0,1,0,0,0,0,
							  0,0,0,1,1,1,0,0,0,
							  0,0,1,1,1,1,1,0,0,
							  0,1,1,1,1,1,1,1,0,
							  1,1,1,1,1,1,1,1,1,
							  0,1,1,1,1,1,1,1,0,
							  0,0,1,1,1,1,1,0,0,
							  0,0,0,1,1,1,0,0,0,
							  0,0,0,0,1,0,0,0,0};

	/*Npp8u ptrMorph_h[121] =	 {0,0,0,0,0,1,0,0,0,0,0,
							  0,0,0,0,1,1,1,0,0,0,0,
							  0,0,0,1,1,1,1,1,0,0,0,
							  0,0,1,1,1,1,1,1,1,0,0,
							  0,1,1,1,1,1,1,1,1,1,0,
							  1,1,1,1,1,1,1,1,1,1,1,
							  0,1,1,1,1,1,1,1,1,1,0,
							  0,0,1,1,1,1,1,1,1,0,0,
							  0,0,0,1,1,1,1,1,0,0,0,
							  0,0,0,0,1,1,1,0,0,0,0,
							  0,0,0,0,0,1,0,0,0,0,0};*/

	// allocate device memory
	//CUDA_SAFE_CALL (cudaMalloc(&ptrInDepth_d, memSizeDepth));
	//CUDA_SAFE_CALL (cudaMalloc(&ptrInColorLab_d, memSizeColor));
	//CUDA_SAFE_CALL (cudaMalloc(&ptrInColorYCrCb_d, memSizeColor));
	//CUDA_SAFE_CALL (cudaMalloc(&ptrBordersDNC_d, memSizeMaskB));
	ptrBordersDNC_d = nppiMalloc_8u_C1 (colsRec, filasRec, &pStepBytes);
	CUDA_SAFE_CALL (cudaMalloc(&ptrInDepthRec_d, memSizeDR));
	//CUDA_SAFE_CALL (cudaMalloc(&ptrInColorRecLab_d, memSizeCR));
	CUDA_SAFE_CALL (cudaMalloc(&ptrInColorRecYCrCb_d, memSizeCR));
	//CUDA_SAFE_CALL (cudaMalloc(&ptrMaskB_d, memSizeMaskB));
	ptrMaskB_d = nppiMalloc_8u_C1 (colsRec, filasRec, &pStepBytes);
	ptrMorph_d = nppiMalloc_8u_C1 (tamMorph, tamMorph, &pStepBytesMorph);
	CUDA_SAFE_CALL (cudaMalloc(&ptrFB_d, memSizeDR));
	CUDA_SAFE_CALL (cudaMalloc(&ptrFBH_d, memSizeDR));
	CUDA_SAFE_CALL (cudaMalloc(&ptrFiltered_d, memSizeDR));
	CUDA_SAFE_CALL (cudaMalloc(&ptrFilteredH_d, memSizeDR));
	CUDA_SAFE_CALL (cudaMalloc(&ptrRellena_d, memSizeDR));
	CUDA_SAFE_CALL (cudaMalloc(&ptrModeloDepth_d, memSizeDR));
	//CUDA_SAFE_CALL (cudaMalloc(&ptrFinalDetection_d, memSizeMaskB));
	ptrFinalDetection_d = nppiMalloc_8u_C1 (colsRec, filasRec, &pStepBytes);
	CUDA_SAFE_CALL (cudaMalloc(&ptrModeloDF_d, memSizeDR));
	CUDA_SAFE_CALL (cudaMalloc(&ptrModeloColor_d, memSizeCR));
	CUDA_SAFE_CALL (cudaMalloc(&ptrRellena2_d, memSizeDR));
	CUDA_SAFE_CALL (cudaMalloc(&ptrProcessedH_d, memSizeDR));
	CUDA_SAFE_CALL (cudaMalloc(&ptrProcessed_d, memSizeDR));
	ptrOutColor_d = nppiMalloc_8u_C1 (colsRec, filasRec, &pStepBytes);
	ptrFilledDetection_d = nppiMalloc_8u_C1 (colsRec, filasRec, &pStepBytes);
	ptrErodedDetection_d = nppiMalloc_8u_C1 (colsRec, filasRec, &pStepBytes);
	ptrBorders_d = nppiMalloc_8u_C1 (colsRec, filasRec, &pStepBytes);
	ptrBordersColor_d = nppiMalloc_8u_C1 (colsRec, filasRec, &pStepBytes);

	cudaMemcpy(ptrMorph_d, ptrMorph_h, memSizeNPPMorph, cudaMemcpyHostToDevice);
	// Load Final Detection of the foreground to device memory initialized to black
	CUDA_SAFE_CALL (cudaMemcpy(ptrFinalDetection_d, ptrFinalDetection_h, memSizeMaskB, cudaMemcpyHostToDevice));
	// Load special mask for borders initialized to black
	CUDA_SAFE_CALL (cudaMemcpy(ptrBordersDNC_d, ptrBordersDNC_h, memSizeMaskB, cudaMemcpyHostToDevice));
	// Load general borders mask initialized to black
	CUDA_SAFE_CALL (cudaMemcpy(ptrMaskB_d, ptrMaskB_h, memSizeMaskB, cudaMemcpyHostToDevice));

	// allocate host memory for sobel operator H & V
	//Matrix sobelH_h;
	Matrix2 sobelH_h;
	sobelH_h.height= SOBEL_SIZE; sobelH_h.width= SOBEL_SIZE;
    unsigned int size_sobel = SOBEL_SIZE * SOBEL_SIZE;
    //size_t memSizeSobel = sizeof(float) * size_sobel;
	size_t memSizeSobel = sizeof(int) * size_sobel;
	//sobelH_h.elements = (float*)malloc(memSizeSobel);
	sobelH_h.elements = (int*)malloc(memSizeSobel);
	createSobelH(sobelH_h.elements);

	Matrix2 sobelV_h;
	sobelV_h.height= SOBEL_SIZE; sobelV_h.width= SOBEL_SIZE;
	sobelV_h.elements = (int*)malloc(memSizeSobel);
	createSobelV(sobelV_h.elements);

	// Load sobel operator to device memory
    Matrix2 sobelH_d;
	sobelH_d.width = sobelH_h.width; sobelH_d.height = sobelH_h.height;
	Matrix2 sobelV_d;
	sobelV_d.width = sobelV_h.width; sobelV_d.height = sobelV_h.height;
	setSobelMask(sobelH_h.elements, sobelV_h.elements, memSizeSobel);

	// allocate host memory for the matrix used as gaussian distance filter
	Matrix gaussDistF_h;
	gaussDistF_h.height= filterSize; gaussDistF_h.width= filterSize;
    unsigned int size_F = filterSize * filterSize;
    size_t memSizeGDF = sizeof(float) * size_F;
	gaussDistF_h.elements = (float*)malloc(memSizeGDF);
	filterInit(gaussDistF_h.elements, filterSize, sigmaS);

	float* elements1D_h;
	size_t memSizeGF1D = sizeof(float) * FILTER_SIZE;
	elements1D_h = (float*)malloc(memSizeGF1D);
	filterInit1D(elements1D_h, filterSize, sigmaS);

	// Load gaussian distance filter to device memory (constant memory)
    Matrix gaussDistF_d;
	gaussDistF_d.width = gaussDistF_h.width; gaussDistF_d.height = gaussDistF_h.height;
	setDistanceMask(gaussDistF_h.elements, memSizeGDF);

	setDistanceMask1D(elements1D_h, memSizeGF1D);

	//int from_to[] = { 0,0,  1,1,  2,2};

	// precalculation of the exp(...) for the final filling
	float* precalculation;
	size_t memSizePrecalculation = sizeof(float) * 256;
	precalculation = (float*)malloc(memSizePrecalculation);
	pre_calculation(precalculation, 256, sigmaCR);
	// Load precalculation to device memory (constant memory)
	setPre_calculation(precalculation, memSizePrecalculation);

	// precalculation of the sigma for the depth related noise
	float* sigmaR;
	size_t memSizeSigmaR = sizeof(float) * 2048;
	sigmaR = (float*)malloc(memSizeSigmaR);
	pre_sigmaR(sigmaR, 2048, aa, bb, cc);
	// Load precalculation to device memory (constant memory)
	setPre_sigmaR(sigmaR, memSizeSigmaR);
	setPre_sigmaR2(sigmaR, memSizeSigmaR);

	cudaArray* colorArray;
	cudaArray* CmapArray;

    MixtureOfGaussians<3, uchar4> mixColor (&mogconfigColor, &parametersColor, &colorArray);
    MixtureOfGaussians<1, unsigned short> mixDepth (&mogconfigDepth, &parametersDepth, &colorArray, &CmapArray);

	setTextureRellenado(&parametersDepth, &CmapArray);

	vector<float> frameTimeVector;
	vector<float> transferTimeVector;
	vector<float> MoGColorTimeVector;
	vector<float> MoGDepthTimeVector;
	vector<float> detectFGErodeVector;
	vector<float> detectFGDiff1Vector;
	vector<float> detectFGAndVector;
	vector<float> detectFGOrVector;
	vector<float> detectFGDiff2Vector;
	vector<float> rellenadoPrevioVector;
	vector<float> detectBordersVector;
	vector<float> detectBordersDilateVector;
	vector<float> rellenadoBordesDNCVector;
	vector<float> medianFilterVector;
	vector<float> bilateralFilterVector;
	vector<float> rellenadoFinalVector;
	vector<float> MoGDepthActualizeVector;
	vector<float> transferBackTimeVector;

	float elapsedTime;

	for (int sequenceNumber = ioconfigDepth.first; sequenceNumber <= ioconfigDepth.last; ++sequenceNumber) {
		#ifdef MEASURETIME
			// capture the start time
			cudaEvent_t start, stop;
			CUDA_SAFE_CALL (cudaEventCreate( &start ));
			CUDA_SAFE_CALL (cudaEventCreate( &stop ));
			CUDA_SAFE_CALL (cudaEventRecord( start, 0 ));
		#endif

		const QString filenameColor = QString().sprintf(ioconfigColor.filenameTemplate.toUtf8().data(), sequenceNumber);
		inColor = imread(ioconfigColor.inputDir.absoluteFilePath (filenameColor).toStdString().c_str());
		//inColor = imread(ioconfigColor.inputDir.absoluteFilePath ("3_colores.JPG").toStdString().c_str());
		//inColor = imread(ioconfigColor.inputDir.absoluteFilePath ("mas_blanco.bmp").toStdString().c_str());
        
		const QString filenameDepth = QString().sprintf(ioconfigDepth.filenameTemplate.toUtf8().data(), sequenceNumber);
		inDepth = imread(ioconfigDepth.inputDir.absoluteFilePath (filenameDepth).toStdString().c_str(), CV_LOAD_IMAGE_ANYDEPTH);
		//sprintf(fileDepthTXT,"%s%d%s", path.c_str(), sequenceNumber, sufDepthS.c_str());
		//conversion(fileDepthTXT, inColor.rows, inColor.cols, inDepth);

		// Converting to Lab color space
		//cvtColor(inColor, inColorLab, CV_BGR2YCrCb);
		cvtColor(inColor, inColorYCrCb, CV_BGR2YCrCb);

		Mat ROIDepth = inDepth(Rect(initCols, initFilas, colsRec, filasRec));
		Mat ROIColor = inColorYCrCb(Rect(initCols, initFilas, colsRec, filasRec));
		//Mat ROIColor = inColor(Rect(initCols, initFilas, colsRec, filasRec));
		ROIDepth.copyTo(inDepthRec);
		ROIColor.copyTo(inColorRecYCrCb);

		/*sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Resultados/Recortada/img_", sequenceNumber, ".Jpeg");
		imwrite(tmpPath, inColorRecYCrCb);
		sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Resultados/Recortada/depth_", sequenceNumber, ".png");
		imwrite(tmpPath, inDepthRec);*/

		#ifdef MEASURETIME2
			// capture the start time
			cudaEvent_t start2, stop2;
			CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
			CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
			CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
		#endif

		// Load images to device memory
		//CUDA_SAFE_CALL (cudaMemcpy(ptrInDepth_d, inDepth.ptr<unsigned short int>(), memSizeDepth, cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL (cudaMemcpy(ptrInColorLab_d, inColorLab.ptr<uchar>(), memSizeColor, cudaMemcpyHostToDevice));
		//CUDA_SAFE_CALL (cudaMemcpy(ptrInColorYCrCb_d, inColorYCrCb.ptr<uchar>(), memSizeColor, cudaMemcpyHostToDevice));
		
		// Load images to device memory
		CUDA_SAFE_CALL (cudaMemcpy(ptrInColorRecYCrCb_d, ptrInColorRecYCrCb_h, memSizeCR, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL (cudaMemcpy(ptrInDepthRec_d, ptrInDepthRec_h, memSizeDR, cudaMemcpyHostToDevice));
		
		// Invoke GPU kernel. To cut the borders of the original images
		//recortar(ptrInDepth_d, ptrInColorLab_d, ptrInDepthRec_d, ptrInColorRecLab_d, initFilas, initCols, filasRec, colsRec, nc, nch);
		//recortar(ptrInDepth_d, ptrInColorYCrCb_d, ptrInDepthRec_d, ptrInColorRecYCrCb_d, initFilas, initCols, filasRec, colsRec, nc, nch);	

		#ifdef MEASURETIME2
			// get stop time, and display the timing results
			CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
			CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
			//float elapsedTime;
			CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
			transferTimeVector.push_back(elapsedTime);
			//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
			//CUDA_SAFE_CALL (cudaEventDestroy( start2 ));
			//CUDA_SAFE_CALL (cudaEventDestroy( stop2 ));
		#endif

		#ifdef MEASURETIME2
			// capture the start time
			//cudaEvent_t start2, stop2;
			CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
			CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
			CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
		#endif
		//CUDA_SAFE_CALL (cudaMemcpy(ptrInDepthRec_h, ptrInDepthRec_d, memSizeDR, cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL (cudaMemcpy(ptrInColorRecLab_h, ptrInColorRecLab_d, memSizeCR, cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL (cudaMemcpy(ptrInColorRecYCrCb_h, ptrInColorRecYCrCb_d, memSizeCR, cudaMemcpyDeviceToHost));

		assert ((inColorRecYCrCb.cols == static_cast<unsigned int>(parametersColor.width))
			&& (inColorRecYCrCb.rows == static_cast<unsigned int>(parametersColor.height)));
        assert ((inDepthRec.cols == static_cast<unsigned int>(parametersDepth.width))
                && (inDepthRec.rows == static_cast<unsigned int>(parametersDepth.height)));

		//mixChannels(&inColorRecYCrCb, 1, &inColor32, 1, from_to, 3);

		mixColor.processImage(&inColor32, &outColor, ptrInColorRecYCrCb_d, ptrModeloColor_d, ptrOutColor_d, thWeightMin, thSigmaMax);
		//CUDA_SAFE_CALL (cudaMemcpy(ptrOutColor_h, ptrOutColor_d, memSizeMaskB, cudaMemcpyDeviceToHost));
		//imshow("deteccion color", outColor);
		//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa3/Deteccion/color/deteccionColor_", sequenceNumber, ".bmp");
		//imwrite(tmpPath, outColor);

		#ifdef MEASURETIME2
			// get stop time, and display the timing results
			CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
			CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
			CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
			//elapsedTime += 0.6f;
			MoGColorTimeVector.push_back(elapsedTime);
			//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
			//CUDA_SAFE_CALL (cudaEventDestroy( start2 ));
			//CUDA_SAFE_CALL (cudaEventDestroy( stop2 ));
		#endif

		//CUDA_SAFE_CALL (cudaMemcpy(ptrModeloColor_h, ptrModeloColor_d, memSizeCR, cudaMemcpyDeviceToHost));
		//CUDA_SAFE_CALL (cudaMemcpy(ptrModeloDepth_h, ptrModeloDepth_d, memSizeDR, cudaMemcpyDeviceToHost));

		//cvtColor(modeloColor, modeloColor, CV_YCrCb2BGR);
		//cvtColor(modeloColor, modeloColor, CV_YCrCb2BGR);
		//cvtColor(modeloColor, modeloColor, CV_BGR2Lab);

		// Process of the first frame
		if (sequenceNumber == ioconfigDepth.first){

			//mixDepth.processImage(&inDepthRec, &outDepth, &finalDetection, ptrModeloDepth_d, aa, bb, cc, deltaSigmaDepth);
			
			// Load Final Detection of the foreground to device memory
			//CUDA_SAFE_CALL (cudaMemcpy(ptrFinalDetection_d, ptrFinalDetection_h, memSizeMaskB, cudaMemcpyHostToDevice));
			// initialized to black
			//CUDA_SAFE_CALL (cudaMemcpy(ptrBordersDNC_d, ptrBordersDNC_h, memSizeMaskB, cudaMemcpyHostToDevice));
			
			// Invoke GPU kernel. To create a Mask for the borders of the objects in the scene
			detectarBordes(sobelH_d, sobelV_d, ptrInDepthRec_d, ptrInColorRecYCrCb_d, ptrMaskB_d, filasRec, colsRec, nch, filterSize, thDepthB, thColorB);

			// copy result from device to host, dilate the mask and copy the result from host to device
			//CUDA_SAFE_CALL (cudaMemcpy(ptrMaskB_h, ptrMaskB_d, memSizeMaskB, cudaMemcpyDeviceToHost));
			//imshow("bordes", mascaraBordes);
			//waitKey();
			//dilate(mascaraBordes, mascaraBordes, elementDilate);
			nppiDilate_8u_C1R (ptrMaskB_d, nStepBytes, &ptrMaskB_d[tamMorph/2*colsRec + tamMorph/2], nStepBytes, oSizeROIRec, ptrMorph_d, oMaskSize, oAnchor);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrMaskB_h, ptrMaskB_d, memSizeMaskB, cudaMemcpyDeviceToHost));
			//imshow("bordes dilatados", mascaraBordes);
			//waitKey();
			//CUDA_SAFE_CALL (cudaMemcpy(ptrMaskB_d, ptrMaskB_h, memSizeMaskB, cudaMemcpyHostToDevice));

			// Invoke GPU kernel. median filter over de uncertainty area
			imageMedianFilter(ptrInDepthRec_d, ptrInColorRecYCrCb_d, ptrInDepthRec_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFB_d, filasRec, colsRec, nch, thFiltroM);
			//imageMedianFilterAprox(ptrInDepthRec_d, ptrInColorRecYCrCb_d, ptrInDepthRec_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFB_d, filasRec, colsRec, nch, thFiltroM);
			//imageMedianFilterH(ptrInDepthRec_d, ptrInColorRecYCrCb_d, ptrInDepthRec_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFBH_d, filasRec, colsRec, nch, thFiltroM);
			//imageMedianFilterV(ptrFBH_d, ptrInColorRecYCrCb_d, ptrInDepthRec_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFB_d, filasRec, colsRec, nch, thFiltroM);

			//cudaMemcpy(ptrFB_h, ptrFB_d, memSizeDR, cudaMemcpyDeviceToHost);
			// Invoke GPU kernel. Bilateral filtering over the rest of the image, with borders already filtered.
			imageBilateralFilter(ptrFB_d, ptrFB_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFiltered_d, filasRec, colsRec, aa, bb, cc);
			//imageBilateralFilterH(ptrFB_d, ptrFB_d, ptrMaskB_d, ptrFinalDetection_d, ptrBordersDNC_d, ptrFilteredH_d, filasRec, colsRec, aa, bb, cc);
			//imageBilateralFilterV(ptrFilteredH_d, ptrFilteredH_d, ptrMaskB_d, ptrFinalDetection_d, ptrBordersDNC_d, ptrFiltered_d, filasRec, colsRec, aa, bb, cc);
			//cudaMemcpy(ptrFiltered_h, ptrFiltered_d, memSizeDR, cudaMemcpyDeviceToHost);

			// Invoke GPU kernel. Fill the final image
			imageRellenar(gaussDistF_d, ptrFiltered_d, ptrInColorRecYCrCb_d, ptrInColorRecYCrCb_d, ptrFinalDetection_d, ptrModeloDepth_d, filasRec, colsRec, nch, sigmaCR, porcentaje, cMin);
			//imageRellenarH(gaussDistF_d, ptrFiltered_d, ptrInColorRecYCrCb_d, ptrInColorRecYCrCb_d, ptrFinalDetection_d, ptrProcessedH_d, filasRec, colsRec, nch, sigmaCR, porcentaje, cMin);
			//imageRellenarV(gaussDistF_d, ptrProcessedH_d, ptrInColorRecYCrCb_d, ptrInColorRecYCrCb_d, ptrFinalDetection_d, ptrModeloDepth_d, filasRec, colsRec, nch, sigmaCR, porcentaje, cMin);
			
			//CUDA_SAFE_CALL (cudaMemcpy(ptrModeloDepth_h, ptrModeloDepth_d, memSizeDR, cudaMemcpyDeviceToHost));
			mixDepth.processImage(&modeloDepth, &outDepth, &finalDetection, ptrModeloDepth_d, ptrModeloDepth_d, aa, bb, cc, deltaSigmaDepth, thWeightMin, thSigmaMax, cMin);

			//modeloColor = inColorRec;
			//inColorRecYCrCb.copyTo(modeloColor);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrModeloColor_d, ptrModeloColor_h, memSizeCR, cudaMemcpyHostToDevice));

			//CUDA_SAFE_CALL (cudaMemcpy(ptrModeloDepth_h, ptrModeloDepth_d, memSizeDR, cudaMemcpyDeviceToHost));
			//imshow("imagen inicial", 15*inDepthRec);
			//imshow("modelo de profundidad", 15*modeloDepth);
			//cv::waitKey();

			#ifdef MEASURETIME
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop ));
				//float elapsedTime;
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start, stop ));
				firstFrameTime = elapsedTime;
				frameTimeVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				CUDA_SAFE_CALL (cudaEventDestroy( start ));
				CUDA_SAFE_CALL (cudaEventDestroy( stop ));
			#endif

			//CUDA_SAFE_CALL (cudaMemcpy(ptrModeloDepth_h, ptrModeloDepth_d, memSizeDR, cudaMemcpyDeviceToHost));
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa2/variasParaAntonio/edgesGROUNDTRUTH/P1/processed_", sequenceNumber, ".png");
			//imwrite(tmpPath, modeloDepth);

		// Processing of the rest of frames
		} else{
			//CUDA_SAFE_CALL (cudaMemcpy(ptrInDepthRec_h, ptrInDepthRec_d, memSizeDR, cudaMemcpyDeviceToHost));
			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start3, stop3;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif

			mixDepth.processImageOnlyDetection(&inDepthRec, &outDepth, &filledDetection, ptrInDepthRec_d, ptrFilledDetection_d, aa, bb, cc, deltaSigmaDepth);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrFilledDetection_h, ptrFilledDetection_d, memSizeMaskB, cudaMemcpyDeviceToHost));
			//imshow("deteccion depth rellena", filledDetection);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa3/Deteccion/profundidad/deteccionDepth_", sequenceNumber, ".bmp");
			//imwrite(tmpPath, filledDetection);

			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				MoGDepthTimeVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start3 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop3 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start4, stop4;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			// Borders detection
			//erode(filledDetection, erodedDetection, elementErode);
			nppiErode_8u_C1R (ptrFilledDetection_d, nStepBytes, &ptrErodedDetection_d[tamMorph/2*colsRec + tamMorph/2], nStepBytes, oSizeROIRec, ptrMorph_d, oMaskSize, oAnchor);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrErodedDetection_h, ptrErodedDetection_d, memSizeMaskB, cudaMemcpyDeviceToHost));
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				detectFGErodeVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start4 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop4 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start4, stop4;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			//borders = filledDetection - erodedDetection;
			nppiAbsDiff_8u_C1R (ptrFilledDetection_d, nStepBytes, ptrErodedDetection_d, nStepBytes, ptrBorders_d, nStepBytes, oSizeROI);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrBorders_h, ptrBorders_d, memSizeMaskB, cudaMemcpyDeviceToHost));
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				detectFGDiff1Vector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start4 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop4 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start4, stop4;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			// Depth borders in color detection
			//bitwise_and(borders, outColor, bordersColor);
			nppiAnd_8u_C1R (ptrBorders_d, nStepBytes, ptrOutColor_d, nStepBytes, ptrBordersColor_d, nStepBytes, oSizeROI);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrBordersColor_h, ptrBordersColor_d, memSizeMaskB, cudaMemcpyDeviceToHost));
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				detectFGAndVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start4 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop4 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start4, stop4;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			// Final detection
			//bitwise_or(bordersColor, erodedDetection, finalDetection);
			nppiOr_8u_C1R (ptrBordersColor_d, nStepBytes, ptrErodedDetection_d, nStepBytes, ptrFinalDetection_d, nStepBytes, oSizeROI);
			CUDA_SAFE_CALL (cudaMemcpy(ptrFinalDetection_h, ptrFinalDetection_d, memSizeMaskB, cudaMemcpyDeviceToHost));
			//imshow("deteccion final", finalDetection);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa3/Deteccion/final/deteccionFinal_", sequenceNumber, ".bmp");
			//imwrite(tmpPath, finalDetection);
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				detectFGOrVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start4 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop4 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start4, stop4;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			//bordersDepthNoColor = filledDetection - finalDetection;
			nppiAbsDiff_8u_C1R (ptrFilledDetection_d, nStepBytes, ptrFinalDetection_d, nStepBytes, ptrBordersDNC_d, nStepBytes, oSizeROI);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrBordersDNC_d, ptrBordersDNC_h, memSizeMaskB, cudaMemcpyHostToDevice));
			//CUDA_SAFE_CALL (cudaMemcpy(ptrBordersDNC_h, ptrBordersDNC_d, memSizeMaskB, cudaMemcpyDeviceToHost));
			// Load Final Detection of the foreground to device memory
			//CUDA_SAFE_CALL (cudaMemcpy(ptrFinalDetection_d, ptrFinalDetection_h, memSizeMaskB, cudaMemcpyHostToDevice));
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				detectFGDiff2Vector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start4 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop4 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start5, stop5;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			// Invoke GPU kernel. Fill the depth image and model before filtering with info from the model and depth image (only BG)
			//rellenarBeforeFiltering(ptrInDepthRec_d, ptrModeloDepth_d, ptrFinalDetection_d, ptrBordersDNC_d, ptrRellena_d, ptrModeloDF_d, filasRec, colsRec);
			rellenarBeforeFiltering(ptrInDepthRec_d, ptrModeloDepth_d, ptrFinalDetection_d, ptrRellena_d, ptrModeloDF_d, filasRec, colsRec);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrRellena_h, ptrRellena_d, memSizeDR, cudaMemcpyDeviceToHost));
			//CUDA_SAFE_CALL (cudaMemcpy(ptrModeloDF_h, ptrModeloDF_d, memSizeDR, cudaMemcpyDeviceToHost));
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				rellenadoPrevioVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start5 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop5 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start6, stop6;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			// Invoke GPU kernel. To create a Mask for the borders of the objects in the scene
			detectarBordes(sobelH_d, sobelV_d, ptrRellena_d, ptrInColorRecYCrCb_d, ptrMaskB_d, filasRec, colsRec, nch, filterSize, thDepthB, thColorB);
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				detectBordersVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start6 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop6 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start6, stop6;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			// copy result from device to host, dilate the mask and copy the result from host to device
			//CUDA_SAFE_CALL (cudaMemcpy(ptrMaskB_h, ptrMaskB_d, memSizeMaskB, cudaMemcpyDeviceToHost));
			//imshow("bordes", mascaraBordes);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Resultados/MascaraBordes/refinada/mascaraBordes_", sequenceNumber, ".bmp");
			//imwrite(tmpPath, mascaraBordes);
			//dilate(mascaraBordes, mascaraBordes, elementDilate);
			//nppiDilate_8u_C1R (ptrMaskB_d, nStepBytes, ptrMaskB_d, nStepBytes, oSizeROI, ptrMorph_d, oMaskSize, oAnchor);
			nppiDilate_8u_C1R (ptrMaskB_d, nStepBytes, &ptrMaskB_d[tamMorph/2*colsRec + tamMorph/2], nStepBytes, oSizeROIRec, ptrMorph_d, oMaskSize, oAnchor);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrMaskB_h, ptrMaskB_d, memSizeMaskB, cudaMemcpyDeviceToHost));
			//imshow("bordes dilatados", mascaraBordes);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Resultados/MascaraBordes/dilatada/mascaraBordesDepth_", sequenceNumber, ".bmp");
			//imwrite(tmpPath, mascaraBordes);
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				detectBordersDilateVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start6 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop6 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start7, stop7;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			// Invoke GPU kernel. To fill with the model the pixels of the borders of the objects detected as background
			rellenarBordersDepthNoColor(ptrRellena_d, ptrModeloDF_d, ptrMaskB_d, ptrBordersDNC_d, ptrRellena2_d, filasRec, colsRec);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrRellena2_h, ptrRellena2_d, memSizeDR, cudaMemcpyDeviceToHost));

			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				rellenadoBordesDNCVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start7 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop7 ));
			#endif


			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start8, stop8;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			// Invoke GPU kernel. median filter over de uncertainty area
			imageMedianFilter(ptrRellena2_d, ptrModeloColor_d, ptrModeloDF_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFB_d, filasRec, colsRec, nch, thFiltroM);
			//imageMedianFilterAprox(ptrRellena2_d, ptrModeloColor_d, ptrModeloDF_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFB_d, filasRec, colsRec, nch, thFiltroM);
			//imageMedianFilterH(ptrRellena2_d, ptrModeloColor_d, ptrModeloDF_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFBH_d, filasRec, colsRec, nch, thFiltroM);
			//imageMedianFilterV(ptrFBH_d, ptrModeloColor_d, ptrModeloDF_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFB_d, filasRec, colsRec, nch, thFiltroM);
			//CUDA_SAFE_CALL (cudaMemcpy(ptrFB_h, ptrFB_d, memSizeDR, cudaMemcpyDeviceToHost));

			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				medianFilterVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start8 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop8 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start9, stop9;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			// Invoke GPU kernel. Bilateral filtering over the rest of the image, with borders already filtered.
			imageBilateralFilter(ptrFB_d, ptrModeloDF_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFiltered_d, filasRec, colsRec, aa, bb, cc);
			//imageBilateralFilterH(ptrFB_d, ptrModeloDF_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFilteredH_d, filasRec, colsRec, aa, bb, cc);
			//imageBilateralFilterV(ptrFilteredH_d, ptrModeloDF_d, ptrMaskB_d, ptrBordersDNC_d, ptrFinalDetection_d, ptrFiltered_d, filasRec, colsRec, aa, bb, cc);
			//cudaMemcpy(ptrFiltered_h, ptrFiltered_d, memSizeDR, cudaMemcpyDeviceToHost);
			//imshow("filtrada", 15*filtered);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Resultados/Filtrado/filtered_", sequenceNumber, ".png");
			//imwrite(tmpPath, filtered);

			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				bilateralFilterVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start9 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop9 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start10, stop10;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			imageRellenar(gaussDistF_d, ptrFiltered_d, ptrInColorRecYCrCb_d, ptrModeloColor_d, ptrFinalDetection_d, ptrProcessed_d, filasRec, colsRec, nch, sigmaCR, porcentaje, cMin);
			//imageRellenarH(gaussDistF_d, ptrFiltered_d, ptrInColorRecYCrCb_d, ptrModeloColor_d, ptrFinalDetection_d, ptrProcessedH_d, filasRec, colsRec, nch, sigmaCR, porcentaje, cMin);
			//imageRellenarV(gaussDistF_d, ptrProcessedH_d, ptrInColorRecYCrCb_d, ptrModeloColor_d, ptrFinalDetection_d, ptrProcessed_d, filasRec, colsRec, nch, sigmaCR, porcentaje, cMin);
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				rellenadoFinalVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				//CUDA_SAFE_CALL (cudaEventDestroy( start10 ));
				//CUDA_SAFE_CALL (cudaEventDestroy( stop10 ));
			#endif

			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start11, stop11;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif
			//cudaMemcpy(ptrProcessed_h, ptrProcessed_d, memSizeDR, cudaMemcpyDeviceToHost);

			mixDepth.processImage(&processed, &outDepth, &filledDetection, ptrProcessed_d, ptrModeloDepth_d, aa, bb, cc, deltaSigmaDepth, thWeightMin, thSigmaMax, cMin);
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				MoGDepthActualizeVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				CUDA_SAFE_CALL (cudaEventDestroy( start2 ));
				CUDA_SAFE_CALL (cudaEventDestroy( stop2 ));
			#endif


			#ifdef MEASURETIME2
				// capture the start time
				//cudaEvent_t start11, stop11;
				CUDA_SAFE_CALL (cudaEventCreate( &start2 ));
				CUDA_SAFE_CALL (cudaEventCreate( &stop2 ));
				CUDA_SAFE_CALL (cudaEventRecord( start2, 0 ));
			#endif

			//CUDA_SAFE_CALL (cudaMemcpy(ptrModeloColor_d, ptrModeloColor_h, memSizeCR, cudaMemcpyHostToDevice));
			cudaMemcpy(ptrProcessed_h, ptrProcessed_d, memSizeDR, cudaMemcpyDeviceToHost);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Resultados/Rellenado/filled_", sequenceNumber, ".png");
			//imwrite(tmpPath, processed);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Resultados/Aproximacion/aprox_", sequenceNumber, ".png");
			//imwrite(tmpPath, processed);
			
			#ifdef MEASURETIME2
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop2, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop2 ));
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start2, stop2 ));
				transferBackTimeVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				CUDA_SAFE_CALL (cudaEventDestroy( start2 ));
				CUDA_SAFE_CALL (cudaEventDestroy( stop2 ));
			#endif

			

			//CUDA_SAFE_CALL (cudaMemcpy(ptrModeloColor_h, ptrModeloColor_d, memSizeCR, cudaMemcpyDeviceToHost));
			//CUDA_SAFE_CALL (cudaMemcpy(ptrModeloDepth_h, ptrModeloDepth_d, memSizeDR, cudaMemcpyDeviceToHost));
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Resultados/ModeloDepth/modelDepth_", sequenceNumber, ".png");
			//imwrite(tmpPath, modeloDepth);
			CUDA_SAFE_CALL (cudaMemcpy(ptrModeloDF_h, ptrModeloDF_d, memSizeDR, cudaMemcpyDeviceToHost));
			//CUDA_SAFE_CALL (cudaMemcpy(ptrFinalDetection_h, ptrFinalDetection_d, memSizeMaskB, cudaMemcpyDeviceToHost));

			// Comprobación
			//cvtColor(modeloColor, modeloColor, CV_YCrCb2BGR);
			cvtColor(inColorRecYCrCb, inColorRecYCrCb, CV_YCrCb2BGR);
			//namedWindow("Image");
			imshow("imagen inicial profundidad", 15*inDepthRec);

			//imshow("imagen inicial RGB", inColorRecYCrCb);
			//imshow("Deteccion", finalDetection);
			//imshow("imagen rellena con modelo", 15*rellena);
			imshow("modelo de profundidad",15*modeloDF);
			imshow("imagen procesada", 15*processed);
			//imshow("modelo de profundidad", 15*modeloDepth);
			//
			imshow("imagen inicial color", inColorRecYCrCb);
			//imshow("bordes filtrados", bordersFiltered);
			//imshow("filtrada", filtered);
			//imshow("rellena", rellena2);
			//waitKey();
			
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa4/genSeqRec/depth_", sequenceNumber, ".png");
			//imwrite(tmpPath, inDepthRec);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa4/genSeqRec/img_", sequenceNumber, ".Jpeg");
			//imwrite(tmpPath, inColorRecYCrCb);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Desktop/modeloDepth/modeloDepth_", sequenceNumber, ".png");
			//imwrite(tmpPath, modeloDepth);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa4/Proceso general/procesadas/processed_", sequenceNumber, ".png");
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa4/Proceso general/procesadasAprox/processed_", sequenceNumber, ".png");
			//imwrite(tmpPath, processed);
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa2/variasParaAntonio/edgesGROUNDTRUTH/boxesFar2/processed_", sequenceNumber, ".png");
			//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa2/variasParaAntonio/edgesGROUNDTRUTH/P1/processed_", sequenceNumber, ".png");
			//imwrite(tmpPath, processed);


			#ifdef MEASURETIME
				// get stop time, and display the timing results
				CUDA_SAFE_CALL (cudaEventRecord( stop, 0 ));
				CUDA_SAFE_CALL (cudaEventSynchronize( stop ));
				//float elapsedTime;
				CUDA_SAFE_CALL (cudaEventElapsedTime( &elapsedTime, start, stop ));
				totalTime += elapsedTime;
				frameTimeVector.push_back(elapsedTime);
				//printf( "	Time to generate: %3.1f ms\n", elapsedTime );
				CUDA_SAFE_CALL (cudaEventDestroy( start ));
				CUDA_SAFE_CALL (cudaEventDestroy( stop ));
			#endif
			waitKey(5);
		}
		//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa2/Ground Truth/Newcombiner_plusC/fgColor_", sequenceNumber, ".bmp");
		//imwrite(tmpPath, outColor);
		//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa2/Ground Truth/Newcombiner_plusD/fgDepth_", sequenceNumber, ".bmp");
		//imwrite(tmpPath, outDepth);
		//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa2/Ground Truth/Newcombiner_plusE/fgEnsamble_", sequenceNumber, ".bmp");
		//imwrite(tmpPath, finalDetection);
	//	imwrite(ioconfigColor.outputDir.absoluteFilePath(filenameColor + sufDetect).toStdString().c_str(), outColor);
	//	imwrite(ioconfigDepth.outputDir.absoluteFilePath(filenameDepth + sufDetect).toStdString().c_str(), outDepth);
		//imwrite(ioconfigDepth.outputDir.absoluteFilePath(filenameDepth + "6filled.tif").toStdString().c_str(), filledDetection);
		//imwrite(ioconfigDepth.outputDir.absoluteFilePath(filenameDepth + "7eroded.tif").toStdString().c_str(), erodedDetection);
	//	const QString filenameEnsamble = QString().sprintf(ioconfigEnsamble.filenameTemplate.toUtf8().data(), sequenceNumber);
		//imwrite(ioconfigEnsamble.outputDir.absoluteFilePath(filenameEnsamble + "8borders.Jpeg").toStdString().c_str(), borders);
		//imwrite(ioconfigEnsamble.outputDir.absoluteFilePath(filenameEnsamble + "9bordersColor.tif").toStdString().c_str(), bordersColor);
	//	imwrite(ioconfigEnsamble.outputDir.absoluteFilePath(filenameEnsamble + sufDetect).toStdString().c_str(), finalDetection);
    }
	//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa4/Calidad/Parametros/Modelado/nModos/modeloDepth5Modos_", 990, ".png");
	//imwrite(tmpPath, modeloDF);
	//sprintf(tmpPath, "%s%d%s", "C:/Users/abb/Documents/Etapa4/Calidad/Parametros/Modelado/nModos/modeloColor5Modos_", 990, ".png");
	//imwrite(tmpPath, modeloColor);

#ifdef MEASURETIME
	meanTime = totalTime / (lastFrame - firstFrame);
	//meanTime = totalTime / 150;
	printf( "****Time to generate first frame: %3.1f ms****\n", firstFrameTime );
	printf( "****Time to generate: %3.1f ms****\n", meanTime );

	string tmp=path+"frameTime.txt";
	writeVector(tmp.c_str(), frameTimeVector);
#endif
#ifdef MEASURETIME2
	tmp=path+"transferTime.txt";
	writeVector(tmp.c_str(), transferTimeVector);
	tmp=path+"MoGColorTime.txt";
	writeVector(tmp.c_str(), MoGColorTimeVector);
	tmp=path+"MoGDepthTime.txt";
	writeVector(tmp.c_str(), MoGDepthTimeVector);
	tmp=path+"detectFGErodeTime.txt";
	writeVector(tmp.c_str(), detectFGErodeVector);
	tmp=path+"detectFGDiff1Time.txt";
	writeVector(tmp.c_str(), detectFGDiff1Vector);
	tmp=path+"detectFGAndTime.txt";
	writeVector(tmp.c_str(), detectFGAndVector);
	tmp=path+"detectFGOrTime.txt";
	writeVector(tmp.c_str(), detectFGOrVector);
	tmp=path+"detectFGDiff2Time.txt";
	writeVector(tmp.c_str(), detectFGDiff2Vector);
	tmp=path+"rellenadoPrevioTime.txt";
	writeVector(tmp.c_str(), rellenadoPrevioVector);
	tmp=path+"detectBordersTime.txt";
	writeVector(tmp.c_str(), detectBordersVector);
	tmp=path+"detectBordersDilateTime.txt";
	writeVector(tmp.c_str(), detectBordersDilateVector);
	tmp=path+"rellenarBordesDNCTime.txt";
	writeVector(tmp.c_str(), rellenadoBordesDNCVector);
	tmp=path+"medianFilterTime.txt";
	writeVector(tmp.c_str(), medianFilterVector);
	tmp=path+"bilateralFilterTime.txt";
	writeVector(tmp.c_str(), bilateralFilterVector);
	tmp=path+"rellenadoFinalTime.txt";
	writeVector(tmp.c_str(), rellenadoFinalVector);
	tmp=path+"MoGDepthActualizeTime.txt";
	writeVector(tmp.c_str(), MoGDepthActualizeVector);
	tmp=path+"transferBackTime.txt";
	writeVector(tmp.c_str(), transferBackTimeVector);
#endif

	frameTimeVector.clear();
	transferTimeVector.clear();
	MoGColorTimeVector.clear();
	MoGDepthTimeVector.clear();
	detectFGErodeVector.clear();
	detectFGDiff1Vector.clear();
	detectFGAndVector.clear();
	detectFGOrVector.clear();
	detectFGDiff2Vector.clear();
	rellenadoPrevioVector.clear();
	detectBordersVector.clear();
	detectBordersDilateVector.clear();
	rellenadoBordesDNCVector.clear();
	medianFilterVector.clear();
	bilateralFilterVector.clear();
	rellenadoFinalVector.clear();
	MoGDepthActualizeVector.clear();
	transferBackTimeVector.clear();

	waitKey();
	//getchar();

	//CUDA_SAFE_CALL (cudaFree(ptrInDepth_d));
	//CUDA_SAFE_CALL (cudaFree(ptrInColorLab_d));
	//CUDA_SAFE_CALL (cudaFree(ptrInColorYCrCb_d));
	CUDA_SAFE_CALL (cudaFree(ptrInDepthRec_d));	
	//CUDA_SAFE_CALL (cudaFree(ptrInColorRecLab_d));
	CUDA_SAFE_CALL (cudaFree(ptrInColorRecYCrCb_d));
	//CUDA_SAFE_CALL (cudaFree(ptrMaskB_d));
	CUDA_SAFE_CALL (cudaFree(ptrFB_d));
	CUDA_SAFE_CALL (cudaFree(ptrFBH_d));
	CUDA_SAFE_CALL (cudaFree(ptrFiltered_d));
	CUDA_SAFE_CALL (cudaFree(ptrFilteredH_d));
	CUDA_SAFE_CALL (cudaFree(ptrRellena_d));
	CUDA_SAFE_CALL (cudaFree(ptrModeloDepth_d));
	//CUDA_SAFE_CALL (cudaFree(ptrFinalDetection_d));
	CUDA_SAFE_CALL (cudaFree(ptrModeloDF_d));
	CUDA_SAFE_CALL (cudaFree(ptrModeloColor_d));
	//CUDA_SAFE_CALL (cudaFree(ptrBordersDNC_d));
	CUDA_SAFE_CALL (cudaFree(ptrRellena2_d));
	CUDA_SAFE_CALL (cudaFree(ptrProcessed_d));

	nppiFree(ptrMaskB_d);
	nppiFree(ptrMorph_d);
	nppiFree(ptrFinalDetection_d);
	nppiFree(ptrBordersDNC_d);
	nppiFree(ptrOutColor_d);
	nppiFree(ptrFilledDetection_d);
	nppiFree(ptrErodedDetection_d);
	nppiFree(ptrBorders_d);
	nppiFree(ptrBordersColor_d);
	
	/**********RELEASE OPENCV IMAGES**********/
	inDepth.release();
	inColor.release();
	//inColorLab.release();
	inColorYCrCb.release();
	inDepthRec.release();
	//inColorRecLab.release();
	inColorRecYCrCb.release();
	inColor32.release();
	outColor.release();
	outDepth.release();
	filledDetection.release();
	erodedDetection.release();
	borders.release();
	bordersColor.release();
	finalDetection.release();
	bordersDepthNoColor.release();
	mascaraBordes.release();
	bordersFiltered.release();
	bordersFilteredH.release();
	filtered.release();
	filteredH.release();
	rellena.release();
	modeloDepth.release();
	modeloDF.release();
	modeloColor.release();
	rellena2.release();
	processed.release();
	morphOperator.release();

	return 0;
}


