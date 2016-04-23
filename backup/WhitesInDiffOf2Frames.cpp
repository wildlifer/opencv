#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
IplImage* background;

long threshouldWhiteDifference = 10;
long noisyWhitesThreshould = 2;
void AllocateImages(IplImage* I){


}
int ObjectDetected(){

	return 1;
}
bool Checkbuffer(long initW, long currentW){
	if ((initW - currentW) < noisyWhitesThreshould)
		return false;
}
int main(int argc, char** argv)
{
	CvCapture* capture = cvCreateFileCapture("videos/2.avi");
	long initWhites = 0;
	IplImage* newFrame = cvQueryFrame(capture);
	background = cvCreateImage(cvSize(newFrame->width, newFrame->height), IPL_DEPTH_8U, 3);
	IplImage*  frame = cvCreateImage(cvSize(newFrame->width, newFrame->height), IPL_DEPTH_8U, 3);
	IplImage* diffFrame;
	IplImage* firstFrame;
	IplImage* secondFrame;
	IplImage* firstFrameGray;
	IplImage* secondFrameGray;
	CvSize sz = cvGetSize(frame);

	diffFrame = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	firstFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	secondFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	firstFrameGray = cvCreateImage((sz), IPL_DEPTH_8U, 1);
	secondFrameGray = cvCreateImage((sz), IPL_DEPTH_8U, 1);

	int b = 0;
	bool first = true;
	bool second = true;
	long initwhites = 0;
	long whites = 0;
	long whitesPrevFrame = 0;
	cvNamedWindow("Original Video", 0);
	cvResizeWindow("Original Video", 400, 400);

	cvNamedWindow("Diff Frame", 0);
	cvResizeWindow("Diff Frame", 400, 400);
	bool buffer = true;
	int pause = 0;
	firstFrame = cvQueryFrame(capture);
	cvCvtColor(firstFrame, firstFrameGray, CV_BGR2GRAY);
	while (1){
		secondFrame = cvQueryFrame(capture);
		cvCvtColor(secondFrame, secondFrameGray, CV_BGR2GRAY);
		cvAbsDiff(firstFrameGray, secondFrameGray, diffFrame);
		cout << " Diff is " << cvCountNonZero(diffFrame) << endl;
		cvShowImage("Original Video", firstFrame);
		cvShowImage("Diff Frame", diffFrame);
		cvCopy(secondFrameGray, firstFrameGray);
		cvWaitKey(30);
	}
		
	int a = cvWaitKey(0);
	cvReleaseCapture(&capture);
	return 0;
}