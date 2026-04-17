#include "mat.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

#include <E:/github_upload/boslab-v2/Common/CUDA/types_BOSLAB.h>
#include <E:/github_upload/boslab-v2/Common/CUDA/GpuIds.hpp>
#include <E:/github_upload/boslab-v2/Common/CUDA/Siddon_projection.h>
#include <E:/github_upload/boslab-v2/Common/CUDA/ray_interpolated_projection.h>
#include <E:/github_upload/boslab-v2/Common/CUDA/BOSLAB_common.h>

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
	geo.TCam = new float[3 * geo.numCam];
	geo.RrCam = new float[9 * geo.numCam];
	geo.TrCam = new float[3 * geo.numCam];
	

	tmp = mxGetField(geoStruct, 0, "nCam");
	geo.nCamU = (int*)malloc(geo.numCam * sizeof(int));
	geo.nCamV = (int*)malloc(geo.numCam * sizeof(int));
	double* nCam = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.nCamU[i] = static_cast<int>(nCam[i * 2]);
		geo.nCamV[i] = static_cast<int>(nCam[i * 2 +1]);
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
	for (int i = 0; i < 9 * geo.numCam; i++) {
		geo.RCam[i] = static_cast<float>(RCam[i]);
	}

	tmp = mxGetField(geoStruct, 0, "TCam");
	geo.TCam = (float*)malloc(3*geo.numCam * sizeof(float));
	double* TCam = mxGetPr(tmp);
	for (int i = 0; i < 3*geo.numCam; i++) {
		geo.TCam[i] = static_cast<float>(TCam[i]);
	}

	tmp = mxGetField(geoStruct, 0, "RrCam");
	geo.RrCam = (float*)malloc(9 * geo.numCam * sizeof(float));
	double* RrCam = mxGetPr(tmp);
	for (int i = 0; i < 9 * geo.numCam; i++) {
		geo.RrCam[i] = static_cast<float>(RrCam[i]);
	}

	tmp = mxGetField(geoStruct, 0, "TrCam");
	geo.TrCam = (float*)malloc(3*geo.numCam * sizeof(float));
	double* TrCam = mxGetPr(tmp);
	for (int i = 0; i < geo.numCam; i++) {
		geo.TrCam[i] = static_cast<float>(TrCam[i]);
	}	

	
	geo.maxnCamU = *std::max_element(geo.nCamU, geo.nCamU +geo.numCam);
	geo.maxnCamV = *std::max_element(geo.nCamV, geo.nCamV + geo.numCam);

	geo.PbkCornerX = new float[4 * geo.numCam];
	geo.PbkCornerY = new float[4 * geo.numCam];
	geo.PbkCornerZ = new float[4 * geo.numCam];

	tmp = mxGetField(geoStruct, 0, "PbkCorner");
	geo.PbkCornerX = (float*)malloc(4 * geo.numCam * sizeof(float));
	geo.PbkCornerY = (float*)malloc(4 * geo.numCam * sizeof(float));
	geo.PbkCornerZ = (float*)malloc(4 * geo.numCam * sizeof(float));
	double* PbkCorner = mxGetPr(tmp);
	for (int i = 0; i < 4*geo.numCam; i++) {
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
	const char* filename = "Axtestdata.mat";
	std::string fullpath = std::string(filepath) + filename;
	const char* file = fullpath.c_str();
	MATFile* matFile = matOpen(file, "r");
	if (matFile == nullptr) {
		std::cerr << "Error opening .mat file: " << file << std::endl;
		return -1;
	}

	// 5th argument is array of GPU-IDs.
	mxArray* gpuIdsArray = matGetVariable(matFile, "GpuDevices");
	if (gpuIdsArray == nullptr) {
		std::cerr << "Error reading GPU IDs variable" << std::endl;
		matClose(matFile);
		return -1;
	}


	GpuIds gpuids;
	{
		size_t iM = mxGetM(gpuIdsArray);
		if (iM != 1) {
			std::cerr << "Error: GPU IDs array must be a row vector." << std::endl;
			matClose(matFile);
			return -1;
		}
		size_t uiGpuCount = mxGetN(gpuIdsArray);
		if (uiGpuCount == 0) {
			std::cerr << "Error: GPU IDs array must contain at least one element." << std::endl;
			matClose(matFile);
			return -1;
		}
		int* piGpuIds = (int*)mxGetData(gpuIdsArray);
		gpuids.SetIds(uiGpuCount, piGpuIds);
	}

	// 4th argument is flow gradient direction in 0 or 1 or 2 or 3
	mxArray* diffselectArray = matGetVariable(matFile, "diffselect");
	if (diffselectArray == nullptr) {
		std::cerr << "Error reading diffselect variable" << std::endl;
		matClose(matFile);
		return -1;
	}

	if (!mxIsDouble(diffselectArray)) {
		std::cerr << "diffselect is not of type double" << std::endl;
		mxDestroyArray(diffselectArray);
		matClose(matFile);
		return -1;
	}

	const int diffselect = static_cast<int>(*mxGetPr(diffselectArray));
	
	// 3th argument is interpolated or Siddon or rk45-interpolated
	mxArray* ptypeArray = matGetVariable(matFile, "ptype");
	if (ptypeArray == nullptr) {
		std::cerr << "Error reading ptype variable" << std::endl;
		matClose(matFile);
		return -1;
	}

	bool rayvoxel = false;
	bool linearBeam = true;
	bool EFtracing = true;
	if (mxIsChar(ptypeArray) != 1) {
		std::cerr << "Error: ptype should be a string." << std::endl;
		matClose(matFile);
		return -1;
	}
	/* copy the string data from prhs[0] into a C string input_ buf.    */
	char* ptype = mxArrayToString(ptypeArray);
	if (strcmp(ptype, "interpolated") != 0 && strcmp(ptype, "Siddon") != 0 && strcmp(ptype, "EF-interpolated") != 0 && strcmp(ptype, "RK-interpolated") != 0) {
		std::cerr << "Error: rayvoxel should be either 'Siddon' or 'interolated' or 'EF-interpolated' or 'RK-interpolated'." << std::endl;
		mxFree(ptype);
		matClose(matFile);
		return -1;
	}
	else {
		// If it's not ray-voxel, it's "interpolated"
		if (strcmp(ptype, "Siddon") == 0) { // strcmp returns 0 if they are equal
			rayvoxel = true;
		}
		else {
			if (strcmp(ptype, "interpolated") == 0) { 
				linearBeam = true;
			}
			else { 
				linearBeam = false; 
				if (strcmp(ptype, "EF-interpolated") == 0) {
					EFtracing = true;
				}
				else if (strcmp(ptype, "RK-interpolated") == 0) {
					EFtracing = false;
				}
			}
		}
	}



	// First input should be x from (Ax=b), or the image.
	mxArray const * const flowArray = matGetVariable(matFile, "flow");
	if (flowArray == nullptr) {
		std::cerr << "Error reading flow variable!" << std::endl;
		matClose(matFile);
		return -1;
	}
	mwSize const  numDims = mxGetNumberOfDimensions(flowArray);

	// Now that input is ok, parse it to C data types.
	float* flow = static_cast<float*>(mxGetData(flowArray));
	// We need a float image, and, unfortunatedly, the only way of casting it is by value
	const mwSize *size_flow = mxGetDimensions(flowArray);



	///////////////////// Second input argument,
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



	printData(numDims, size_flow, geo, diffselect, ptype, gpuids);


	// Create a 3D array for the output
	mxArray *plhs[1];
	mwSize outsize[3];
	outsize[0] = geo.maxnCamV;
	outsize[1] = geo.maxnCamU;
	outsize[2] = geo.numCam;
	
	plhs[0] = mxCreateNumericArray(3, outsize, mxSINGLE_CLASS, mxREAL);
	float *outProjections = static_cast<float *>(mxGetData(plhs[0]));  // We will NOT be freeing this pointer!


	// MODIFICATION, RB, 5/12/2017: As said above, we do not allocate anything, just
	// set pointers in result to point to outProjections
	float** result = (float**)malloc(geo.numCam * sizeof(float*)); // This only allocates memory for pointers
	unsigned long long projSizeInPixels = geo.maxnCamU * geo.maxnCamV;
	for (int i = 0; i < geo.numCam; i++)
	{
		unsigned long long currProjIndex = projSizeInPixels * i;
		result[i] = &outProjections[currProjIndex]; // now the pointers are the same
	}

	// call the real function

	if (rayvoxel) {
		auto start = std::chrono::high_resolution_clock::now();
		siddon_ray_projection(flow, geo, diffselect, result, gpuids);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;
		mexPrintf("Total Time: %f\n", duration.count());
	}
	else {	
		interpolation_projection(flow, geo, diffselect, linearBeam, EFtracing, result, gpuids);

	}



	// Save the result to a .mat file
	filename = "Projection.mat";
	fullpath = std::string(filepath) + filename;
	file = fullpath.c_str();
	saveToMatFile(file, plhs[0], "projections");

	// Free MATLAB array memory
	mxDestroyArray(plhs[0]);


	// Add the call to the real function here, e.g.,
	// myFunction(geo, angles, nangles, coneBeam, gpuIds, modeString);

	matClose(matFile);

	return 0;
}