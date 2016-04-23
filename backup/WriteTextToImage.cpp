#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	Mat vehicle = Mat::zeros(200, 1000, CV_8UC3);
	//addText(vehicle, "Car", Point(50, 50),font("Times", 12, Scalar(255), CV_FONT_BOLD, CV_STYLE_NORMAL, 1));
	putText(vehicle, "Car", Point(vehicle.cols/2-vehicle.cols/10, vehicle.rows/2), FONT_HERSHEY_SIMPLEX,4,Scalar(255,255,255),6,8,false);
	imshow("T",vehicle);
	char k = cvWaitKey(0);
	return 0;
}