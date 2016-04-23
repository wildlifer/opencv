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
	CvCapture* capture = cvCreateFileCapture("videos/2.avi");
	if (!capture) {
		std::cout << "Cant read the video";
		return (-1);
	}
	IplImage* frame = cvQueryFrame(capture);
	IplImage* smallFrame = cvCreateImage(cvSize(frame->width/2, frame->height/2), frame->depth, frame->nChannels);
	IplImage* gray = cvCreateImage(cvSize(frame->width / 2, frame->height / 2), frame->depth, 1);
	cvResize(frame, smallFrame, CV_INTER_LINEAR);
	double fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
	//smallFrame = cvCreateImage(cvSize(frame->width, frame->height), frame->depth, frame->nChannels);
	while ((frame = cvQueryFrame(capture)) != NULL){
		//smallFrame = cvCreateImage(cvSize(600, 600), frame->depth, frame->nChannels);
		cvResize(frame, smallFrame, CV_INTER_LINEAR);
		cvCvtColor(smallFrame, gray, CV_BGR2GRAY);
		cvShowImage("1", gray);
		cvWaitKey(50);
		//cvWriteFrame(writer, frame);
	}

	cvWaitKey(100);
	return 0;
}