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
	cvNamedWindow("Camera_Output", 1);    //Create window
	cvResizeWindow("Camera_Output", 600, 400);
	CvCapture* capture = cvCreateFileCapture("videos/1.5.mp4");
	IplImage* frame = cvQueryFrame(capture);
	//frame = ;
	//IplImage* frameGray = NULL;// cvQueryFrame(capture);//Capture using any camera connected to your system
	IplImage* frameGray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	IplImage* frameBinary = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	cvCvtColor(frame, frameGray, CV_BGR2GRAY);
	cvThreshold(frameGray, frameBinary, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	long initialWhites = cvCountNonZero(frameBinary);
	long newwhites = 0;
	cout << " Initial Whites are " << initialWhites << endl;
	cvWaitKey(1000);
	int inp = 0;
	while (1){ //Create infinte loop for live streaming
		 frame= cvQueryFrame(capture);
		 if (!frame) break;
		 //frameGray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
		 cvCvtColor(frame, frameGray, CV_BGR2GRAY);
		 cvThreshold(frameGray, frameBinary, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		 cvResize(frameGray, frameGray, 3);
		 newwhites = cvCountNonZero(frameBinary);
		 //cout << "Difference Whites are " << initialWhites - newwhites << endl;
		 cout << "No. of Whites are " <<  newwhites << endl;
		 if ((initialWhites - newwhites) > 15){
			 cvShowImage("Camera_Output", frameBinary);
			// cvResizeWindow("Camera_Output", 600, 400);
			
		 }
		 else{
			 cvShowImage("Camera_Output", frameBinary);
			// cvResizeWindow("Camera_Output", 600, 400);
			 //Show image frames on created window
			 cvWaitKey(50);
		 }
		 inp = cvWaitKey(100);
		 if (inp == 32)
			 cvWaitKey(0);
	}
	cvReleaseCapture(&capture); //Release capture.
	cvDestroyWindow("Camera_Output"); //Destroy Window
	cvWaitKey(-1);
	return 0;
}