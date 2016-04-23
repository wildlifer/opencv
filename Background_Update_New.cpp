#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;
using namespace cv;

ofstream outFile;

long whites = 0;
const int LEARN_FRAME_LIMIT = 5;
int frameLearnCounter = 0;
int CHISQUARE = 1;
int CHISQUARE_THRES = 5;
int BHATTA = 3;
int BHATTA_THRES = 3;
int INTERSEC = 2;
int INTERSEC_THRES = 7;

int ORIGIN = 0;
int SHIFT = 350;
int WINDISTANCE = 20;

//crop parameters
double cropY = 0.40;
double cropHeight = 0.30;

//window dispay parameters
int max_rows = 300;
int max_cols = 400;

int main(int argc, char** argv)
{
	VideoCapture cap("videos/4.2.mp4");
	Mat background, hsv_background, gray_background;
	Mat nextframe, hsv_nextframe, gray_nextframe;
	Mat diff_frame;
	outFile.open("data/VehicleCassifier/changeInMetric.txt", 'w');

	cap >> background;
	cap >> background;
	int width = background.cols;
	int height = background.rows;

	int cropFrameY = height*cropY;
	int cropFrameHeight = height*cropHeight;

	Rect roi(0, cropFrameY, width, cropFrameHeight);

	long initWhites = 0;
	
	//background = background(roi);
	cvtColor(background, hsv_background, COLOR_BGR2HSV);
	cvtColor(background, gray_background, COLOR_BGR2GRAY);

	cvNamedWindow("Original Video", CV_WINDOW_NORMAL);
	//cvShowImage("Original Video", hsv_background);
	cvResizeWindow("Original Video", max_cols, max_rows);
	cvMoveWindow("Original Video", ORIGIN+SHIFT, 0);

	/// Using 50 bins for hue and 60 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };
	/// Histograms
	MatND hist_background;
	MatND hist_nextframe;
	MatND store[LEARN_FRAME_LIMIT];

	
	//first frame
	calcHist(&hsv_background, 1, channels, Mat(), hist_background, 2, histSize, ranges, true, false);
	normalize(hist_background, hist_background, 0, 1, NORM_MINMAX, -1, Mat());

	CvSize sz = cvGetSize(new IplImage(background));
	IplImage* oldFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);;
	IplImage* frame		= cvCreateImage(sz, IPL_DEPTH_8U, 3);;
	IplImage* diffFrameBinary = cvCreateImage(sz, IPL_DEPTH_8U, 3);;

	IplImage* agray = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	IplImage* bgray = cvCreateImage(sz, IPL_DEPTH_8U, 1);;

	IplImage* abin = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	IplImage* bbin = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	IplImage* diff = cvCreateImage(sz, IPL_DEPTH_8U, 1);;

	int b = 0;
	bool first = true;
	long initwhites = 0;
	long whites = 0;
	std::vector<KeyPoint> keypoints;

	cvNamedWindow("Diff Frame", 0);
	while (1) {
		//background conversions
		//background = background(roi);
		cvtColor(background, hsv_background, COLOR_BGR2HSV);
		cvtColor(background, gray_background, COLOR_BGR2GRAY);
		calcHist(&hsv_background, 1, channels, Mat(), hist_background, 2, histSize, ranges, true, false);
		normalize(hist_background, hist_background, 0, 1, NORM_MINMAX, -1, Mat());
		//new frame
		cap >> nextframe;
		if (nextframe.empty()) break;
		//get region of interest
		//nextframe = nextframe(roi);
		cvtColor(nextframe, hsv_nextframe, COLOR_BGR2HSV);
		cvtColor(nextframe, gray_nextframe, COLOR_BGR2GRAY);
		calcHist(&hsv_nextframe, 1, channels, Mat(), hist_nextframe, 2, histSize, ranges, true, false);
		normalize(hist_nextframe, hist_nextframe, 0, 1, NORM_MINMAX, -1, Mat());
	
		//Show diff frame  - IplImage model
		oldFrame = new IplImage(background);
		cvCvtColor(oldFrame, agray, CV_BGR2GRAY);
		cvNamedWindow("Current background", CV_WINDOW_NORMAL);
		cvResizeWindow("Current background", max_cols, max_rows);
		cvShowImage("Current background", oldFrame);	
		cvMoveWindow("Current background", ORIGIN + SHIFT + WINDISTANCE + max_cols, 0);
		//cvThreshold(agray, abin, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		frame = new IplImage(nextframe);
		cvCvtColor(frame, bgray, CV_BGR2GRAY);
		cvNamedWindow("Original Video", CV_WINDOW_NORMAL);
		cvResizeWindow("Original Video", max_cols, max_rows);
		cvShowImage("Original Video", frame);
		cvMoveWindow("Original Video", ORIGIN + SHIFT, 0);
		//cvThreshold(bgray, bbin, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

		//Get the difference between background and current frame
		cvAbsDiff(agray, bgray, diff);
		//cvAbsDiff(abin, bbin, diff);
		//absdiff(gray_background, gray_nextframe, diff_frame);

		//Mat Model
		/*absdiff(background, nextframe, diff_frame);*/
		cvNamedWindow("Diff Frame", CV_WINDOW_NORMAL);
		cvResizeWindow("Diff Frame", max_cols, max_rows);
		cvShowImage("Diff Frame", diff);
		cvMoveWindow("Diff Frame", ORIGIN + SHIFT + WINDISTANCE + max_cols, 350);

		//Model parameters
		double changeCHI = compareHist(hist_background, hist_nextframe, CHISQUARE);
		double changeBHATTA = compareHist(hist_background, hist_nextframe, BHATTA);
		double changeINTERSEC = compareHist(hist_background, hist_nextframe, INTERSEC);
		//cout << "Change in Bhatta is " << changeBHATTA << endl;
		//cout << "Change in Chi is " << changeCHI << endl;
		cout << "Change in Intersec is " << changeINTERSEC << endl;
		////New Background
		if (changeINTERSEC < INTERSEC_THRES){
			frameLearnCounter++;
			if (frameLearnCounter == LEARN_FRAME_LIMIT){
				frameLearnCounter = 0;
				nextframe.copyTo(background);
				cout << "Updating Background" << endl;
				int xx;
				cin >> xx;
				cin.clear();
				cin.ignore(INT_MAX, '\n');
			}
		}
		else if (changeINTERSEC>INTERSEC_THRES){
			frameLearnCounter = 0;
		}
		outFile << changeINTERSEC << endl;
		cvWaitKey(30);
	}
	int a = cvWaitKey(100000);
	return 0;
}