# include "de.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace cv;

double evaluate(int D, double tmp[], long *nfeval)
{
	double result = 0;
	Mat vehicle = Mat::zeros(200, 1260, CV_8UC3);
	Mat clearedVehicle = Mat::zeros(200, 1260, CV_8UC3);
	// will hold the results of the detection
	std::vector<Vec3f> circles;
	// runs the actual detection
	//HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 8, cannyThreshold, accumulatorThreshold, 14, 40);
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 14, cannyThreshold, accumulatorThreshold, 14, 35);
	const int noOfCircles = circles.size();
	int noOfFinalCircles = 0;

	return result;
}
