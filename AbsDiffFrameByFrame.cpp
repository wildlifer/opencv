#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

long whites = 0;

int main(int argc, char** argv)
{
	CvCapture* capture = cvCreateFileCapture("videos/4.1.mp4");
	VideoCapture cap("videos/4.1.mp4");
	Mat background, hsv_background, gray_background;
	Mat nextframe, hsv_nextframe, gray_nextframe;
	Mat diff_frame;

	MatND hist_background;
	MatND hist_nextframe;
	/// Using 50 bins for hue and 60 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };

	cap >> background;
	cvtColor(background, hsv_background, COLOR_BGR2HSV);
	cvtColor(background, gray_background, COLOR_BGR2GRAY);
	calcHist(&hsv_background, 1, channels, Mat(), hist_background, 2, histSize, ranges, true, false);
	normalize(hist_background, hist_background, 0, 1, NORM_MINMAX, -1, Mat());

	long initWhites = 0;
	IplImage* newFrame = cvQueryFrame(capture);
	//IplImage*  frame = cvCreateImage(cvSize(newFrame->width / 3.5, newFrame->height / 3), IPL_DEPTH_8U, 3);
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
	std::vector<KeyPoint> keypoints;
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
		//Calculate diff
		cvAbsDiff(oldFrame, frameGray, diffFrame);
		cvSmooth(diffFrame, diffFrame, CV_BLUR);
		cvThreshold(diffFrame, diffFrame, 25, 255, CV_THRESH_BINARY);

		//show images
		cvShowImage("Original Video", frame);
		cvMoveWindow("Original Video", 0, 0);

		
		//cvShowImage("Original Gray", frameGray);
		//cvMoveWindow("Original Gray", 100, 100);
		cvShowImage("Diff Frame", diffFrame);
		cvMoveWindow("Diff Frame", (newFrame->width / 1) + 10, 0);

		//New Background
		//cvConvertScale(frameGray, oldFrame, 1.0, 0.0);
		cvCopy(frameGray, oldFrame, NULL);
		cvShowImage("Current background", oldFrame);
		cvMoveWindow("Current background", 0, 300);
		//oldFrame = frameGray;
		cvWaitKey(10);
	}
	int a = cvWaitKey(100000);
	cvReleaseCapture(&capture);
	return 0;
}