
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <thread>
#include <iomanip>

#include <cv.h>
#include <opencv2/opencv.hpp>
#include <highgui.h>
#include <eigen3/Eigen/Dense>
#include "System.h"

using namespace std;
using namespace cv;
using namespace Eigen;

const int nDelayTimes = 2;
string sData_path;
string sConfig_path;

std::shared_ptr<System> pSystem;

void PubImuData()
{
	string sImu_data_file = sData_path + "imu0/data.csv";
	ifstream fsImu;
	fsImu.open(sImu_data_file.c_str());
	if (!fsImu.is_open())
	{
		cerr << "Failed to open imu file! " << sImu_data_file << endl;
		return;
	}

	std::string sImu_line;
	double dStampNSec = 0.0;
	Vector3d vAcc;
	Vector3d vGyr;
	while (std::getline(fsImu, sImu_line) && !sImu_line.empty()) // read imu data
	{
        if(sImu_line[0]=='#') continue;
        char dot;
		std::istringstream ssImuData(sImu_line);
		ssImuData >> dStampNSec >>dot>> vGyr.x() >>dot>> vGyr.y() >>dot>> vGyr.z() >>dot>> vAcc.x() >> dot>>vAcc.y() >>dot>> vAcc.z();
		pSystem->PubImuData(dStampNSec / 1e9, vGyr, vAcc);
		usleep(5000*nDelayTimes);
	}
	fsImu.close();
}

void PubImageData()
{
	string sImage_file = sData_path + "cam0/data.csv";

	ifstream fsImage;
	fsImage.open(sImage_file.c_str());
	if (!fsImage.is_open())
	{
		cerr << "Failed to open image file! " << sImage_file << endl;
		return;
	}

	std::string sImage_line;
	double dStampNSec;
	string sImgFileName;
	
	// cv::namedWindow("SOURCE IMAGE", CV_WINDOW_AUTOSIZE);
	while (std::getline(fsImage, sImage_line) && !sImage_line.empty())
	{
        if(sImage_line[0]=='#') continue;
		std::istringstream ssImuData(sImage_line);
		ssImuData >> dStampNSec >> sImgFileName;
		sImgFileName.erase(sImgFileName.begin());
		string imagePath = sData_path + "cam0/data/" + sImgFileName;

		Mat img = imread(imagePath.c_str(), 0);
		if (img.empty())
		{
			cerr << "image is empty! path: " << imagePath << endl;
			return;
		}
		pSystem->PubImageData(dStampNSec / 1e9, img);
		usleep(50000*nDelayTimes);
	}
	fsImage.close();
}

int main(int argc, char **argv)
{
	if(argc != 3)
	{
		cerr << "./run_euroc PATH_TO_EUROC/mav0 PATH_TO_CONFIG/config \n"
			<< endl;
		return -1;
	}
	sData_path = argv[1];
	sConfig_path = argv[2];

	pSystem.reset(new System(sConfig_path));
    //std::thread thd_PGOpt(&System::PoseGraphOptimization, pSystem);
	std::thread thd_BackEnd(&System::ProcessBackEnd, pSystem);



	std::thread thd_PubImuData(PubImuData);

	std::thread thd_PubImageData(PubImageData);

	std::thread thd_Draw(&System::Draw, pSystem);



	thd_PubImuData.join();
	thd_PubImageData.join();
    thd_BackEnd.join();
    //thd_PGOpt.join();

sleep(5);

	return 0;
}
