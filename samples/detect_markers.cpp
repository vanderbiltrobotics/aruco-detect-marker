/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <fstream>

#include <covariance-tracker/covariance-tracker.h>
#include <Eigen/Dense>
#include <iostream>
#include <vector>



using namespace std;
using namespace cv;

namespace {
const char* about = "Basic marker detection";
const char* keys  =
        "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
        "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
        "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
        "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
        "{v        |       | Input from video file, if ommited, input comes from camera }"
        "{ci       | 0     | Camera id if input doesnt come from video (-v) }"
        "{c        |       | Camera intrinsic parameters. Needed for camera pose }"
        "{l        | 0.1   | Marker side lenght (in meters). Needed for correct scale in camera pose }"
        "{dp       |       | File of marker detector parameters }"
        "{r        |       | show rejected candidates too }"
	"{t        |       | threshold}";
}

/**
 */
static bool readCameraParameters(string filename, Mat &camMatrix, Mat &distCoeffs) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}



/**
 */
static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params) {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
}



/**
 */
int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv, keys);
    parser.about(about);

    if(argc < 2) {
        parser.printMessage();
        return 0;
    }

    int dictionaryId = parser.get<int>("d");
    bool showRejected = parser.has("r");
    bool estimatePose = parser.has("c");
    float markerLength = parser.get<float>("l");
   

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    if(parser.has("dp")) {
        bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
        if(!readOk) {
            cerr << "Invalid detector parameters file" << endl;
            return 0;
        }
    }
    detectorParams->cornerRefinementMethod = aruco::CORNER_REFINE_SUBPIX; // do corner refinement in markers

    int camId = parser.get<int>("ci");

    String video;
    if(parser.has("v")) {
        video = parser.get<String>("v");
    }

    if(!parser.check()) {
        parser.printErrors();
        return 0;
    }

    Ptr<aruco::Dictionary> dictionary =
        aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    Mat camMatrix, distCoeffs;
    if(estimatePose) {
        bool readOk = readCameraParameters(parser.get<string>("c"), camMatrix, distCoeffs);
        if(!readOk) {
            cerr << "Invalid camera file" << endl;
            return 0;
        }
    }


    VideoCapture inputVideo;
    int waitTime;
    if(!video.empty()) {
        inputVideo.open(video);
        waitTime = 0;
    } else {
        inputVideo.open(camId);
        waitTime = 10;
    }

    //SET CAMERA RESOLUTION --- UNTESTED
 //   inputVideo.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  //  inputVideo.set(CV_CAP_PROP_FRAME_HEIGHT,960);
  
	ofstream myfileDistance2;
	myfileDistance2.open("distance2.txt");
	ofstream myfileDistance1;
	myfileDistance1.open("distance1.txt");
	ofstream myfileDistance0;
	myfileDistance0.open("distance0.txt");
	ofstream myfileRotate0;
	ofstream myfileRotate1;
	ofstream myfileRotate2;
	myfileRotate0.open("rotate0.txt");
	myfileRotate1.open("rotate1.txt");
	myfileRotate2.open("rotate2.txt");

    double totalTime = 0;
    int totalIterations = 0;
// setting up the covariance matrix
	CovarianceTracker<double, 3> cov(50);
	Eigen::Matrix<double, 3, 3> c;
	
	
	
    int di = 1;
	int midThreshold = 50;
    int threshold = 60;
	double rangeMax = threshold + 50;
	double rangeMin = threshold - 50;
	double startTime = 0;
	double timePassed = 0;
	int count = 0;
    while(inputVideo.grab()) {
        Mat image, imageCopy, imageOriginal;
        inputVideo.retrieve(imageOriginal);
	
	cvtColor(imageOriginal,image,CV_BGR2GRAY);
	double min, max;
	cv::minMaxLoc(image, &min, &max);
	max = 255/max;
//	bilateralFilter(imageCopy,image,6,12,3);	
	image = (image-threshold)*255;
	
	cv::minMaxLoc(image, &min, &max);
        double tick = (double)getTickCount();

        vector< int > ids;
        vector< vector< Point2f > > corners, rejected;
        vector< Vec3d > rvecs, tvecs;

        // detect markers and estimate pose
        aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
        if(estimatePose && ids.size() > 0)
            aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs,
                                             tvecs);

        double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
        totalTime += currentTime;
        totalIterations++;
        if(totalIterations % 10 == 0) {
            cout << "Detection Time = " << currentTime * 1000 << " ms "
                 << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)"
		 << endl;
        }
	
        // draw results
	//edit image:
	imageOriginal.copyTo(imageCopy);
        if(ids.size() > 0) {
		cov.addData(tvecs);//This is data covariance tracker
		c = cov.getCovariance;
		startTime = getTickCount();
		cout << "'Start Time' is now: " << startTime/getTickFrequency() << endl;
            aruco::drawDetectedMarkers(imageCopy, corners, ids);
            if(estimatePose) {
                for(unsigned int i = 0; i < ids.size(); i++)
		{
                    aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i],
                                    markerLength * 0.5f);
			

		    cout << "Distance = " << tvecs[i] << endl;
			cout << "Rotation = "<< rvecs[i] << endl;
			cout << "Threshold = "<< min << "  " << max<<endl;
			cout << "Covariance Matrix = "<<c<<endl;
			myfileDistance0 << tvecs[i][0] << endl;
			myfileDistance1 << tvecs[i][1] << endl;
			myfileDistance2 << tvecs[i][2] << endl;
			myfileRotate0 << rvecs[i][0] << endl;
			myfileRotate1 << rvecs[i][1] << endl;
			myfileRotate2 << rvecs[i][2] << endl;
		}
		if (count>2)
			midThreshold = threshold;
		if (rangeMax>(midThreshold+2))
			rangeMax=rangeMax-(rangeMax-midThreshold)/10;
		if (rangeMin<(midThreshold-2))
			rangeMin=rangeMin+(midThreshold-rangeMin)/10;
		
           }
	++count; // Count how many consecutive frames the marker has been seen in.
	cout << "marker detected for " << count << " consecutive frames."<<" Range max:"<<rangeMax << endl;
	}
	else  {
		cout<<"No marker detected"<<endl;
		timePassed = (getTickCount() - startTime)/getTickFrequency(); // Calculate seconds since marker was last detected
		cout << timePassed << " secs passed since " << startTime << endl;		
		if (timePassed >= 1 || count < 10) // Check if marker hasn't been detected in a while, or if was only detected for a short time  
		{ 
			threshold = threshold + di;
			if (rangeMax<100)
                        	rangeMax=rangeMax+0.25;
                	if (rangeMin>0)
                        	rangeMin=rangeMin-0.25;   
			if (threshold > rangeMax || threshold < rangeMin) di = di *(-1);
			cout << "Adjusting threshold:"<< threshold << " threshold range:"<< rangeMax<<"-"<<rangeMin  << endl;
			count = 0;
		}
		else{
			cout << "Waiting: " << endl;
		}
	}

	
        

        if(showRejected && rejected.size() > 0)
            aruco::drawDetectedMarkers(imageCopy, rejected, noArray(), Scalar(100, 0, 255));
	
//        imshow("out", imageCopy);
        char key = (char)waitKey(waitTime);
        if(key == 27) break;
    }
	myfileDistance0.close();
	myfileDistance1.close();
	myfileDistance2.close();
	myfileRotate0.close();
	myfileRotate1.close();
	myfileRotate2.close();
    return 0;
}

