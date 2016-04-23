#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

IplImage* doPyrDown(IplImage* imgIn, int filter = IPL_GAUSSIAN_5x5){
	assert(imgIn->width % 2 == 0 && imgIn->height % 2 == 0);
	IplImage* imgOut = cvCreateImage(
		cvSize(imgIn->width / 2, imgIn->height / 2),
		IPL_DEPTH_8U,
		3);
	cvPyrDown(imgIn, imgOut);
	return (imgOut);
}
IplImage* doCanny(
	IplImage* in,
	double lowThresh,
	double highThresh,
	double aperture)
{
	if (in->nChannels != 1){
		std::cout << "Canny only handls grayscale images";
		return (0);
	}
	IplImage* out = cvCreateImage(
		cvSize(in->width,in->height),
		IPL_DEPTH_8U,
		1
		);
	cvCanny(in, out, lowThresh, highThresh, aperture);
	return out;
}

int main(int argc, char** argv)
{
	IplImage* imgIn = cvLoadImage("images/highway/bus1.png");
	cvNamedWindow("In", CV_WINDOW_AUTOSIZE);
	cvShowImage("In", imgIn);

	cvNamedWindow("Out", CV_WINDOW_AUTOSIZE);
	IplImage* imgOut = doPyrDown(imgIn, IPL_GAUSSIAN_5x5);
	IplImage* imgOutGray = cvCreateImage(cvGetSize(imgOut), IPL_DEPTH_8U, 1);
	cvCvtColor(imgOut, imgOutGray, COLOR_BGR2GRAY);
	IplImage* imgOutCanny = doCanny(imgOutGray, 10, 100, 3);
	cvShowImage("Out", imgOutCanny);

	cvWaitKey(0);
	cvReleaseImage(&imgIn);
	cvDestroyWindow("In");
}