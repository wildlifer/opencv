#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

long whites = 0;
void AllocateImages(IplImage* I){
	
	
}
int main(int argc, char** argv)
{
	CvCapture* capture = cvCreateFileCapture("videos/2.mp4");
	long initWhites = 0;
	IplImage* newFrame = cvQueryFrame(capture);
	IplImage*  frame = cvCreateImage(cvSize(newFrame->width / 3.5, newFrame->height / 3), IPL_DEPTH_8U, 3);
	IplImage* diffFrame;
	IplImage* oldFrame;
	IplImage* frameGray;
	CvSize sz = cvGetSize(frame);

	diffFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	oldFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	frameGray = cvCreateImage((sz), IPL_DEPTH_8U, 1);
	cvResize(newFrame, frame, 3);
	/*IplImage* secondFrame = cvQueryFrame(capture);
	IplImage* thirdFrame = cvQueryFrame(capture);
	
	IplImage*  frame2 = cvCreateImage(cvSize(firstFrame->width / 3.5, firstFrame->height / 3), IPL_DEPTH_8U, 3);
	IplImage*  frame3 = cvCreateImage(cvSize(firstFrame->width / 3.5, firstFrame->height / 3), IPL_DEPTH_8U, 3);
	AllocateImages(frame);
	//get the diff of first and second frame
	cvResize(firstFrame, frame, 3);
	cvResize(secondFrame, frame2, 3);
	cvResize(thirdFrame, frame3, 3);
	cvAbsDiff(frame, frame2, diffFrame);
	cvCvtColor(diffFrame, frameGray, CV_BGR2GRAY);
	//get the whites of first frame difference
	initWhites = cvCountNonZero(frameGray);
	cout << "Init Whites  are " << (initWhites) << endl;
	//
	cvCopy(frame, tempFrame);
	cvAbsDiff(tempFrame, frame3, diffFrame);
	cvCvtColor(diffFrame, frameGray, CV_BGR2GRAY);
	//get the whites of first frame difference
	
	cout << "Init Whites  1 are " << (initWhites) << endl;
	cvCopy(frame, tempFrame,NULL);
	//tempFrame = frame;
	//cvShowImage("tempFrame", tempFrame);
	//cvMoveWindow("tempFrame", (firstFrame->width / 3) + 10, (firstFrame->height / 3) + 10);
	*/
	int b = 0;
	bool first = true;
	long initwhites = 0;
	long whites = 0;
	while (1) {
		newFrame = cvQueryFrame(capture);
		if (!newFrame) break;
		cvResize(newFrame, frame, 3);
		cvCvtColor(frame, frameGray, CV_BGR2GRAY);
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
		cvAbsDiff(oldFrame,frameGray, diffFrame);
		
		cout << "Current Frame whites are " << (whites) << endl;
		cout << "change in whites are " << (whites - initWhites) << endl;
		cvSmooth(diffFrame, diffFrame, CV_BLUR);
		cvThreshold(diffFrame, diffFrame, 25, 255, CV_THRESH_BINARY);
		cvShowImage("Original Video", frame);
		cvMoveWindow("Original Video", 0, 0);
		//Calculate diff
		
		cvShowImage("Diff Frame", diffFrame);
		cvMoveWindow("Diff Frame", (newFrame->width / 3) + 10, 0);
		//New Background
		cvConvertScale(frameGray, oldFrame, 1.0, 0.0);

		cvWaitKey(10);
	}
	int a = cvWaitKey(100000);
	cvReleaseCapture(&capture);
	return 0;
}