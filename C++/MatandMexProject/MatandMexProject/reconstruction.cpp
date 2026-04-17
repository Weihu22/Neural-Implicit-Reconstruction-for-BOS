#include "mat.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <E:/github_upload/boslab-v2/Common/CUDA/types_BOSLAB.h>
#include <E:/github_upload/boslab-v2/Common/CUDA/GpuIds.hpp>
#include <E:/github_upload/boslab-v2/Common/CUDA/gpuUtils.hpp>
#include <E:/github_upload/boslab-v2/Common/CUDA/BOSLAB_common.h>
#include <E:/github_upload/boslab-v2/Common/CUDA/imgtodisplacement.h>
#include <E:/github_upload/boslab-v2/Common/CUDA/reconstruction.h>
#include <E:/github_upload/boslab-v2/Common/CUDA/imgtoreconstruction.h>
#include <E:/github_upload/boslab-v2/Common/CUDA/gpu_buffer_pool.h>
#include <E:/github_upload/boslab-v2/Common/CUDA/uvtoeps.h>

void saveToMatFile(const std::string &filename, mxArray *array, const char *varname) {
	MATFile *matFile = matOpen(filename.c_str(), "w");
	if (matFile == nullptr) {
		std::cerr << "Error opening MAT file: " << filename << std::endl;
		return;
	}
	if (matPutVariable(matFile, varname, array) != 0) {
		std::cerr << "Error saving variable to MAT file." << std::endl;
	}
	matClose(matFile);
}

bool loadAllImagesFromFolder(const std::string& folder,
	std::vector<cv::Mat>& images,
	std::vector<std::string>& imageNames)
{
	images.clear();
	imageNames.clear();

	std::vector<cv::String> files;
	// 匹配文件夹下所有 img*.png / jpg / bmp
	cv::glob(folder + "/*.png", files, false);
	cv::glob(folder + "/*.jpg", files, false);
	cv::glob(folder + "/*.bmp", files, false);

	if (files.empty()) {
		std::cerr << "No images found in: " << folder << std::endl;
		return false;
	}

	// 按文件名排序（按字符串，可改为自然排序）
	std::sort(files.begin(), files.end());

	for (const auto& f : files) {
		cv::Mat img = cv::imread(f, cv::IMREAD_GRAYSCALE);
		if (img.empty()) {
			std::cerr << "Failed to read image: " << f << std::endl;
			return false;
		}
		images.push_back(img);
		imageNames.push_back(f);
	}

	std::cout << "Loaded " << images.size() << " images." << std::endl;
	return true;
}


bool parseGeometry(mxArray* geoStruct, Geometry& geo) {
	mxArray *tmp;

	geo.unitX = 1;
	geo.unitY = 1;
	geo.unitZ = 1;

	mxArray* nVoxelField = mxGetField(geoStruct, 0, "nVoxel");
	mxArray* sVoxelField = mxGetField(geoStruct, 0, "sVoxel");
	mxArray* dVoxelField = mxGetField(geoStruct, 0, "dVoxel");
	mxArray* OprField = mxGetField(geoStruct, 0, "Opr");

	double* nVoxel = mxGetPr(nVoxelField);
	double* sVoxel = mxGetPr(sVoxelField);
	double* dVoxel = mxGetPr(dVoxelField);
	double* Opr = mxGetPr(OprField);

	geo.nVoxelX = static_cast<int>(nVoxel[0]);
	geo.nVoxelY = static_cast<int>(nVoxel[1]);
	geo.nVoxelZ = static_cast<int>(nVoxel[2]);

	geo.sVoxelX = static_cast<float>(sVoxel[0]);
	geo.sVoxelY = static_cast<float>(sVoxel[1]);
	geo.sVoxelZ = static_cast<float>(sVoxel[2]);

	geo.dVoxelX = static_cast<float>(dVoxel[0]);
	geo.dVoxelY = static_cast<float>(dVoxel[1]);
	geo.dVoxelZ = static_cast<float>(dVoxel[2]);

	geo.OprX = static_cast<float>(Opr[0]);
	geo.OprY = static_cast<float>(Opr[1]);
	geo.OprZ = static_cast<float>(Opr[2]);

	mxArray* numCamField = mxGetField(geoStruct, 0, "numCam");
	double* numCam = mxGetPr(numCamField);
	geo.numCam = static_cast<int>(numCam[0]);

	geo.nCamU = new int[geo.numCam];
	geo.nCamV = new int[geo.numCam];
	geo.dCamU = new float[geo.numCam];
	geo.dCamV = new float[geo.numCam];
	geo.sCamU = new float[geo.numCam];
	geo.sCamV = new float[geo.numCam];
	geo.OcrX = new float[geo.numCam];
	geo.OcrY = new float[geo.numCam];
	geo.OcrZ = new float[geo.numCam];
	geo.fCam = new int[geo.numCam];
	geo.Zbc = new float[geo.numCam];
	geo.Zpc = new float[geo.numCam];
	geo.fd = new float[geo.numCam];
	geo.IMCam = new float[9 * geo.numCam];
	geo.RCam = new float[9 * geo.numCam];
	geo.TCam = new float[3*geo.numCam];
	geo.RrCam = new float[9 * geo.numCam];
	geo.TrCam = new float[9*geo.numCam];

	tmp = mxGetField(geoStruct, 0, "nCam");
	geo.nCamU = (int*)malloc(geo.numCam * sizeof(int));
	geo.nCamV = (int*)malloc(geo.numCam * sizeof(int));
	double* nCam = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.nCamU[i] = static_cast<int>(nCam[i * 2]);
		geo.nCamV[i] = static_cast<int>(nCam[i * 2 + 1]);
	}

	tmp = mxGetField(geoStruct, 0, "dCam");
	geo.dCamU = (float*)malloc(geo.numCam * sizeof(float));
	geo.dCamV = (float*)malloc(geo.numCam * sizeof(float));
	double* dCam = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.dCamU[i] = static_cast<float>(dCam[i * 2]);
		geo.dCamV[i] = static_cast<float>(dCam[i * 2 + 1]);
	}

	tmp = mxGetField(geoStruct, 0, "sCam");
	geo.sCamU = (float*)malloc(geo.numCam * sizeof(float));
	geo.sCamV = (float*)malloc(geo.numCam * sizeof(float));
	double* sCam = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.sCamU[i] = static_cast<float>(sCam[i * 2]);
		geo.sCamV[i] = static_cast<float>(sCam[i * 2 + 1]);
	}

	tmp = mxGetField(geoStruct, 0, "Ocr");
	geo.OcrX = (float*)malloc(geo.numCam * sizeof(float));
	geo.OcrY = (float*)malloc(geo.numCam * sizeof(float));
	geo.OcrZ = (float*)malloc(geo.numCam * sizeof(float));
	double* Ocr = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.OcrX[i] = static_cast<float>(Ocr[i * 3]);
		geo.OcrY[i] = static_cast<float>(Ocr[i * 3 + 1]);
		geo.OcrZ[i] = static_cast<float>(Ocr[i * 3 + 2]);
	}

	tmp = mxGetField(geoStruct, 0, "fCam");
	geo.fCam = (int*)malloc(geo.numCam * sizeof(int));
	double* fCam = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.fCam[i] = static_cast<int>(fCam[i]);
	}

	tmp = mxGetField(geoStruct, 0, "Zbc");
	geo.Zbc = (float*)malloc(geo.numCam * sizeof(float));
	double* Zbc = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.Zbc[i] = static_cast<float>(Zbc[i]);
	}

	tmp = mxGetField(geoStruct, 0, "Zpc");
	geo.Zpc = (float*)malloc(geo.numCam * sizeof(float));
	double* Zpc = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.Zpc[i] = static_cast<float>(Zpc[i]);
	}

	tmp = mxGetField(geoStruct, 0, "fd");
	geo.fd = (float*)malloc(geo.numCam * sizeof(float));
	double* fd = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.fd[i] = static_cast<float>(fd[i]);
	}

	tmp = mxGetField(geoStruct, 0, "IMCam");
	geo.IMCam = (float*)malloc(9 * geo.numCam * sizeof(float));
	double* IMCam = mxGetPr(tmp);
	for (int i = 0; i < 9 * geo.numCam; i++) {
		geo.IMCam[i] = static_cast<float>(IMCam[i]);
	}

	tmp = mxGetField(geoStruct, 0, "RCam");
	geo.RCam = (float*)malloc(9 * geo.numCam * sizeof(float));
	double* RCam = mxGetPr(tmp);
	for (int i = 0; i < 3 * geo.numCam; i++) {
		geo.RCam[i] = static_cast<float>(RCam[i]);
	}

	tmp = mxGetField(geoStruct, 0, "TCam");
	geo.TCam = (float*)malloc(9*geo.numCam * sizeof(float));
	double* TCam = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.TCam[i] = static_cast<float>(TCam[i]);
	}

	tmp = mxGetField(geoStruct, 0, "RrCam");
	geo.RrCam = (float*)malloc(9 * geo.numCam * sizeof(float));
	double* RrCam = mxGetPr(tmp);
	for (int i = 0; i < 9 * geo.numCam; i++) {
		geo.RrCam[i] = static_cast<float>(RrCam[i]);
	}

	tmp = mxGetField(geoStruct, 0, "TrCam");
	geo.TrCam = (float*)malloc(9*geo.numCam * sizeof(float));
	double* TrCam = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.TrCam[i] = static_cast<float>(TrCam[i]);
	}


	geo.maxnCamU = *std::max_element(geo.nCamU, geo.nCamU + geo.numCam);
	geo.maxnCamV = *std::max_element(geo.nCamV, geo.nCamV + geo.numCam);

	geo.PbkCornerX = new float[4 * geo.numCam];
	geo.PbkCornerY = new float[4 * geo.numCam];
	geo.PbkCornerZ = new float[4 * geo.numCam];

	tmp = mxGetField(geoStruct, 0, "PbkCorner");
	geo.PbkCornerX = (float*)malloc(4 * geo.numCam * sizeof(float));
	geo.PbkCornerY = (float*)malloc(4 * geo.numCam * sizeof(float));
	geo.PbkCornerZ = (float*)malloc(4 * geo.numCam * sizeof(float));
	double* PbkCorner = mxGetPr(tmp);
	for (int i = 0; i < 4 * geo.numCam; i++) {
		geo.PbkCornerX[i] = static_cast<float>(PbkCorner[i * 3]);
		geo.PbkCornerY[i] = static_cast<float>(PbkCorner[i * 3 + 1]);
		geo.PbkCornerZ[i] = static_cast<float>(PbkCorner[i * 3 + 2]);
	}

	geo.accuracy = static_cast<float>(mxGetScalar(mxGetField(geoStruct, 0, "accuracy")));
	tmp = mxGetField(geoStruct, 0, "accuracy");
	double* acc = mxGetPr(tmp);
	if (acc[0] < 0.001)
		std::cerr << "Accuracy should be bigger than 0.001" << std::endl;

	geo.accuracy = (float)acc[0];


	return true;

}

std::vector<int> readGpuIds(mxArray* gpuIdsArray) {
	std::vector<int> gpuIds;
	if (mxIsInt32(gpuIdsArray) || mxIsUint32(gpuIdsArray)) {
		int* piGpuIds = static_cast<int*>(mxGetData(gpuIdsArray));
		size_t numIds = mxGetNumberOfElements(gpuIdsArray);
		gpuIds.assign(piGpuIds, piGpuIds + numIds);
	}
	else {
		std::cerr << "Error: GPU IDs should be an integer array." << std::endl;
	}
	return gpuIds;
}


int main() {
	const char* filepath = "E:/github_upload/boslab-v2/MATLAB/Test_data/hotgun_flow/";
	const char* filename = "before_reconstruction_data.mat";
	std::string fullpath = std::string(filepath) + filename;
	const char* file = fullpath.c_str();
	MATFile* matFile = matOpen(file, "r");
	if (matFile == nullptr) {
		std::cerr << "Error opening .mat file: " << file << std::endl;
		return -1;
	}
	
	// Geometry structure that has all the needed geometric data.
	mxArray* geoStruct = matGetVariable(matFile, "geo");
	if (geoStruct == nullptr || !mxIsStruct(geoStruct)) {
		std::cerr << "Error reading geo Struct" << std::endl;
		matClose(matFile);
		return -1;
	}

	Geometry geo;
	if (!parseGeometry(geoStruct, geo)) {
		matClose(matFile);
		return -1;
	}

	printGeoData(geo);

	// img obtained from file
	/*std::string imgFolder = std::string(filepath) + "parallelcomputing/iref/";

	std::vector<cv::Mat> iref;
	std::vector<cv::Mat> idis;
	std::vector<std::string> imageNames;

	if (!loadAllImagesFromFolder(imgFolder, iref, imageNames)) { return -1;}	
	std::cout << "First iref image size: " << iref[0].cols << " x " << iref[0].rows << std::endl;

	imgFolder = std::string(filepath) + "parallelcomputing/idis/";
	if (!loadAllImagesFromFolder(imgFolder, idis, imageNames)) {return -1;}
	std::cout << "First idis image size: " << idis[0].cols << " x " << iref[0].rows << std::endl;*/
	
	
	// set gpuids 
	GpuIds gpuids;
	{	
		int iCudaDeviceCount = GetGpuCount();
		if (iCudaDeviceCount == 0) {
			mexPrintf("Error: No CUDA-capable GPUs found!\n");
			return -1;
		}
		int* gpuIdArray = new int[iCudaDeviceCount];
		char message[65535];
		int numDevices = GetGpuIdArray("", gpuIdArray, iCudaDeviceCount, message);
		mexPrintf("GpuIds message: %s\n", message);

		if (numDevices == 0) {
			mexPrintf("Error: No matching GPUs found!\n");
			delete[] gpuIdArray;
			return -1;
		}

		gpuids.SetIds(numDevices, gpuIdArray);
	
	}

	ReconstructionPara reconP;
	reconP.lambda = 10;
	reconP.niter = 40;
	reconP.niter_outer = 3;
	reconP.niter_break = std::round(reconP.niter / reconP.niter_outer);

	GPUBufferPool gpuPool(geo);
	gpuPool.allocate(geo, reconP.niter);

	// calculate displacement
	//mxArray *plhsdispu[1];
	//mxArray *plhsdispv[1];
	//mwSize outsizedisp[3];
	//outsizedisp[0] = geo.maxnCamV;
	//outsizedisp[1] = geo.maxnCamU;
	//outsizedisp[2] = geo.numCam;
	//plhsdispu[0] = mxCreateNumericArray(3, outsizedisp, mxSINGLE_CLASS, mxREAL);
	//plhsdispv[0] = mxCreateNumericArray(3, outsizedisp, mxSINGLE_CLASS, mxREAL);
	//float *projectionsU = (float *)mxGetPr(plhsdispu[0]);
	//float *projectionsV = (float *)mxGetPr(plhsdispv[0]);

	//imgtodisplacement(geo, filepath, gpuids, projectionsU, projectionsV, gpuPool);

	//mxArray *plhsprojeps[1];
	//mwSize outsizedispop[3];
	//outsizedispop[0] = geo.maxnCamV;
	//outsizedispop[1] = geo.maxnCamU;
	//outsizedispop[2] = 3*geo.numCam;
	//plhsprojeps[0] = mxCreateNumericArray(3, outsizedispop, mxSINGLE_CLASS, mxREAL);
	//float *projectionsob = (float *)mxGetPr(plhsprojeps[0]);

	//uvtoeps(geo, projectionsU, projectionsV, projectionsob, gpuids, gpuPool);

	//// save the result to a.mat file
	//filename = "flowDispu.mat";
	//fullpath = std::string(filepath) + filename;
	//file = fullpath.c_str();
	//saveToMatFile(file, plhsdispu[0], "flowDispu");

	//filename = "flowDispv.mat";
	//fullpath = std::string(filepath) + filename;
	//file = fullpath.c_str();
	//saveToMatFile(file, plhsdispv[0], "flowDispv");

	//// save the result to a .mat file
	//filename = "projectionsob.mat";
	//fullpath = std::string(filepath) + filename;
	//file = fullpath.c_str();
	//saveToMatFile(file, plhsprojeps[0], "projectionsob");

	//// create a 3d array for the output
	mxArray *plhs[1];
	mwSize outsize[3];
	outsize[0] = geo.nVoxelX;
	outsize[1] = geo.nVoxelY;
	outsize[2] = geo.nVoxelZ;
	plhs[0] = mxCreateNumericArray(3, outsize, mxSINGLE_CLASS, mxREAL);
	float *flowRecon = (float *)mxGetPr(plhs[0]);
	//


	//IRN_TV_CGLS(geo, projectionsU, projectionsV, reconP, flowRecon, gpuids, gpuPool);
	img_to_reconstruction(geo, filepath, gpuids, flowRecon,reconP, gpuPool);
	gpuPool.freeAll();
	


	filename = "flowRecon.mat";
	fullpath = std::string(filepath) + filename;
	file = fullpath.c_str();
	saveToMatFile(file, plhs[0], "flowRecon");


	// Free MATLAB array memory
	/*mxDestroyArray(plhs[0]);*/

	// Add the call to the real function here, e.g.,
	// myFunction(geo, angles, nangles, coneBeam, gpuIds, modeString);

	matClose(matFile);

	return 0;
}