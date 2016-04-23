#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
IplImage* background;
long whites = 0;
void AllocateImages(IplImage* I){


}
int main(int argc, char** argv)
{
	CvCapture* capture = cvCreateFileCapture("videos/1.2.mp4");
	long initWhites = 0;
	IplImage* newFrame = cvQueryFrame(capture);
	background = cvCreateImage(cvSize(newFrame->width, newFrame->height), IPL_DEPTH_8U, 3);
	IplImage*  frame = cvCreateImage(cvSize(newFrame->width, newFrame->height), IPL_DEPTH_8U, 3);
	IplImage* diffFrame;
	IplImage* oldFrame;
	IplImage* frameGray;
	CvSize sz = cvGetSize(frame);

	diffFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	oldFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	frameGray = cvCreateImage((sz), IPL_DEPTH_8U, 1);
	cvResize(newFrame, frame, 3);
	
	int b = 0;
	bool first = true;
	long initwhites = 0;
	long whites = 0;
	long whitesPrevFrame = 0;
	cvNamedWindow("Original Video", 0);
	cvResizeWindow("Original Video", 400, 400);

	cvNamedWindow("Diff Frame", 0);
	cvResizeWindow("Diff Frame", 400, 400);
	bool noise = true;
	while (1) {
		newFrame = cvQueryFrame(capture);
		if (!newFrame) break;
		//cvResize(newFrame, frame, 3);
		cvCvtColor(newFrame, frameGray, CV_BGR2GRAY);
		if (first){
			diffFrame = cvCloneImage(frameGray);
			oldFrame = cvCloneImage(frameGray);
			first = false;
			initWhites = cvCountNonZero(frameGray);
			cvConvertScale(frameGray, oldFrame, 1.0, 0.0);
			cout << "Init whites are " << (initWhites) << endl;
			continue;
		}
		whites = cvCountNonZero(frameGray);
		cvAbsDiff(oldFrame, frameGray, diffFrame);

		cout << "Current Frame whites are " << (whites) << endl;
		cout << "change in whites are " << (initWhites -whites ) << endl;
		if ((initWhites - whites)<2){
			noise = false;
		}
		if ((initWhites - whites)>10 && noise == false){
			cout << "change detected" <<endl;
			int c = cvWaitKey(3000);
			noise = true;
		}
		cvSmooth(diffFrame, diffFrame, CV_BLUR);
		cvThreshold(diffFrame, diffFrame, 25, 255, CV_THRESH_BINARY);
		cvShowImage("Original Video", newFrame);
		cvMoveWindow("Original Video", 0, 0);
		//Calculate diff

		cvShowImage("Diff Frame", diffFrame);
		cvMoveWindow("Diff Frame", (newFrame->width / 3) + 10, 0);
		//New Background
		cvConvertScale(frameGray, oldFrame, 1.0, 0.0);
		whitesPrevFrame = whites;
		cvWaitKey(1);
	}
	int a = cvWaitKey(100000);
	cvReleaseCapture(&capture);
	return 0;
}