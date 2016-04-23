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
	double fps = cvGetCaptureProperty(capture,CV_CAP_PROP_FPS);
	CvSize size = cvSize(
		(int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH),
		(int)cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT)
		);
	CvVideoWriter *writer = cvCreateVideoWriter(
		"videos/copied.avi",
		CV_FOURCC('P', 'I', 'M', '1'),
		30,
		cvGetSize(frame)
		);
	IplImage* logpolar_frame = cvCreateImage(
		cvGetSize(frame),
		IPL_DEPTH_8U,
		1
		);
	while ((frame=cvQueryFrame(capture)) !=NULL){
		//cvLogPolar(	frame,logpolar_frame,
			//	cvPoint2D32f(frame->width,frame->height),
				//	40.0,
					//CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS
					//);
		//cvCvtColor(frame, logpolar_frame, CV_BGR2GRAY);
		cvWriteFrame(writer, logpolar_frame);
		//cvShowImage("Def", logpolar_frame);
		//cvWaitKey(10);
	}
		cvWaitKey(100);
	return 0;
}