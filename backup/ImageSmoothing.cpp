#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	IplImage* imgIn = cvLoadImage("images/highway/bus1.png");
	cvNamedWindow("In", CV_WINDOW_AUTOSIZE);
	cvShowImage("In",imgIn);

	cvNamedWindow("Out",CV_WINDOW_AUTOSIZE);
	IplImage* imgOut = cvCreateImage(
						cvGetSize(imgIn),
						IPL_DEPTH_8U,
						3);
	cvSmooth(imgIn,imgOut,CV_GAUSSIAN,3,3);
	cvShowImage("Out",imgOut);

	cvWaitKey(0);
	cvReleaseImage(&imgIn);
	cvDestroyWindow("In");
}