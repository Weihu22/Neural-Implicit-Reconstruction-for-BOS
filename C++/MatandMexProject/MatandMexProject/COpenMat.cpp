#include <iostream>
#include "BOSLABreconstruct.h"

// Function to call 'defaultGeometry' and print or handle the geometry
void initializeGeometry() {
	// Initialize geometry using the given parameters
	mwArray geo;

	// Arguments: nVoxel, sVoxel, angles, nCam, dCam, fCam, Zbc, Zpc
	mwArray nVoxel(3, 1, mxDOUBLE_CLASS);  // 3x1 array for voxel counts
	mwArray sVoxel(3, 1, mxDOUBLE_CLASS);  // 3x1 array for voxel sizes
	mwArray angles(10, 1, mxDOUBLE_CLASS); // Array for angles (10 elements)
	mwArray nCam(2, 1, mxDOUBLE_CLASS);    // Camera parameters
	mwArray dCam(2, 1, mxDOUBLE_CLASS);    // Camera pixel size
	mwArray fCam(1, 1, mxDOUBLE_CLASS);    // Focal length
	mwArray Zbc(1, 1, mxDOUBLE_CLASS);    // Background camera distance
	mwArray Zpc(1, 1, mxDOUBLE_CLASS);    // Projector camera distance

	// Fill the arrays with values as in MATLAB code
	nVoxel.SetData(new double[3]{ 44, 66, 44 }, 3);
	sVoxel.SetData(new double[3]{ 22, 33, 22 }, 3);
	angles.SetData(new double[10]{ 0, 18, 36, 54, 72, 90, 108, 126, 144, 162 }, 10);
	nCam.SetData(new double[2]{ 316, 794 }, 2);
	dCam.SetData(new double[2]{ 0.02, 0.02 }, 2);
	fCam.SetData(new double[1]{ 105 }, 1);
	Zbc.SetData(new double[1]{ 775 }, 1);
	Zpc.SetData(new double[1]{ 430 }, 1);

	mwArray str_nVoxel("nVoxel");
	mwArray str_sVoxel("sVoxel");
	mwArray str_angles("angles");
	mwArray str_nCam("nCam");
	mwArray str_dCam("dCam");
	mwArray str_fCam("fCam");
	mwArray str_Zbc("Zbc");
	mwArray str_Zpc("Zpc");

	// Create a cell array to hold the input arguments
	mwArray varargin(1, 16, mxCELL_CLASS);  // Cell array to hold all input arguments

		// Assign each parameter into the respective cell of varargin
	varargin.Get(1, 1).Set(str_nVoxel);
	varargin.Get(1, 2).Set(nVoxel);
	varargin.Get(1, 3).Set(str_sVoxel);
	varargin.Get(1, 4).Set(sVoxel);
	varargin.Get(1, 5).Set(str_angles);
	varargin.Get(1, 6).Set(angles);
	varargin.Get(1, 7).Set(str_nCam);
	varargin.Get(1, 8).Set(nCam);
	varargin.Get(1, 9).Set(str_dCam);
	varargin.Get(1, 10).Set(dCam);
	varargin.Get(1, 11).Set(str_fCam);
	varargin.Get(1, 12).Set(fCam);
	varargin.Get(1, 13).Set(str_Zbc);
	varargin.Get(1, 14).Set(Zbc);
	varargin.Get(1, 15).Set(str_Zpc);
	varargin.Get(1, 16).Set(Zpc);
	// Call the defaultGeometry function with the inputs
	defaultGeometry(1, geo, varargin);  // nargout = 0, geo will hold the output


	// Print the result (or handle as needed)
	std::cout << "Geometry generate successfully!" << std::endl;
	// Optionally, you can access geo or use it further
	// std::cout << geo << std::endl;  // Depending on your needs, print or process geo


		// Print each field of the struct 'geo'
	int numFields = geo.NumberOfFields();
	if (numFields >0) {

		// Iterate over all fields in the struct
		for (size_t i = 0; i < numFields; ++i) {
			mwString fieldName = geo.GetFieldName(i);
			std::cout << "Field name: " << fieldName << std::endl;

			mwArray fieldData = geo.Get(fieldName, 1, 1);
			int n = fieldData.NumberOfElements();
			std::cout << "Field size: " << n << std::endl;
		}
	}
	else {
		std::cout << "Geo is not a struct." << std::endl;
	}
	mwArray h;
	plotgeometry(1, h, geo);
	mclWaitForFiguresToDie(NULL);  //等待图像显示，不加此句无法显示图像

	// Print the result (or handle as needed)
	std::cout << "plotgeometry successfully!" << std::endl;
	// Optionally, you can access geo or use it further
	// std::cout << geo << std::endl;  // Depending on your needs, print or process geo

}

int main() {
	try {
		if (!BOSLABreconstructInitialize()) {
			std::cerr << "Initialize BOSLAB fail" << std::endl;
			return 1;
		}
		else {
			std::cerr << "Initialize BOSLAB success" << std::endl;
		}
		// Initialize geometry
		initializeGeometry();

		// 终止 BOSLAB 库
		BOSLABreconstructTerminate();
	}
	catch (const std::exception &e) {
		std::cerr << "Error: " << e.what() << std::endl;
		BOSLABreconstructTerminate();
		return 1;
	}

	return 0;
}
