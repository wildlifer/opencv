#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
ofstream outFile;

namespace
{
	// windows and trackbars name
	const std::string windowName = "Hough Circle Detection Demo";
	const std::string cannyThresholdTrackbarName = "Canny threshold";
	const std::string accumulatorThresholdTrackbarName = "Accumulator Threshold";
	const std::string inverseResolutionName = "Resolution";
	const std::string usage = "Usage : tutorial_HoughCircle_Demo <path_to_input_image>\n";

	// initial and max values of the parameters of interests.
	const int cannyThresholdInitialValue = 98;
	const int accumulatorThresholdInitialValue = 25;
	const int maxAccumulatorThreshold = 400;
	const int maxCannyThreshold = 255;
	const int inverseResolutionInitialValue = 1;
	const int maxInverseResolution = 3;

	void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold, int inverseResolution)
	{
		// will hold the results of the detection
		std::vector<Vec3f> circles;
		// runs the actual detection
		HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 3, cannyThreshold, accumulatorThreshold, 0, 40);

		/// Apply the Hough Transform to find the circles
		//HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1.98, src_gray.rows / 5, 250, 50, 0, 30);

		// clone the colour, input image for displaying purposes
		Mat display = src_display.clone();
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle center
			circle(display, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			circle(display, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		}
		switch (circles.size()){
		case 1:outFile << " False - " << circles.size() << endl; break;
		case 2:outFile << " Car - " << circles.size() << endl; break;
		case 3:outFile << " Car - " << circles.size() << endl; break;
		case 4:outFile << " Truck - " << circles.size() << endl; break;
		default: outFile << " SpaceShip - " << circles.size() << endl; break;
		}

		// shows the results
		imshow("Original Video", display);
		cvWaitKey(100);
	}
}
IplImage* getBestFrame(map<long, IplImage*> m){
	IplImage* frame = NULL;
	map<long, IplImage*>::iterator it;
	long mean = 0L;
	long key = 0L;
	int i = 0;
	for (it = m.begin(), i = 0; it != m.end(); it++, i++){
		mean += it->first;
		if (i == m.size() / 2){
			key = it->first;
			break;
		}
	}
	mean = (long)mean / m.size();
	cout << "Mean is " << mean << endl;
	cout << "Best found at " << key << " at location "<<i<<endl;
	//cvShowImage("Best1", m[key]);
	return  m[key];
}

int main(int argc, char** argv)
{
	CvCapture* capture = cvCreateFileCapture("videos/1.1.mp4");
	outFile.open("data\\Classifier.txt");

	IplImage* frame = cvQueryFrame(capture);
	IplImage* frameGray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	IplImage* bestFrame = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	cvCvtColor(frame, frameGray, CV_BGR2GRAY);
	long initialWhites = cvCountNonZero(frameGray);
	long whites = 0;
	long thresholdWhitecount = 70;
	int counter = 0;
	long framecounter = 0;
	std::map<long, IplImage*> m;
	while (1) {
		frame = cvQueryFrame(capture);
		if (!frame) break;
		cvCvtColor(frame, frameGray, CV_BGR2GRAY);
		whites = cvCountNonZero(frameGray);
		framecounter++;
		//if ((initialWhites - whites) < thresholdWhitecount) continue;
		if (++counter == 2){
			counter = 0;
			bestFrame = getBestFrame(m);
			//cvShowImage("Best",bestFrame);
			
			//char jj = cvWaitKey(0);
			Mat src(frame), src_gray(bestFrame);
			//cvCvtColor( frameGray, frame,CV_GRAY2BGR);
			// Reduce the noise so we avoid false circle detection
			GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

			//declare and initialize both parameters that are subjects to change
			int cannyThreshold = cannyThresholdInitialValue;
			int accumulatorThreshold = accumulatorThresholdInitialValue;
			int inverseResolution = inverseResolutionInitialValue;

			// create the main window, and attach the trackbars
			cvNamedWindow("Original Video", 0);
			cvResizeWindow("Original Video", 600, 600);

			//cvResizeWindow(windowName, 400, 400);
			createTrackbar(cannyThresholdTrackbarName, windowName, &cannyThreshold, maxCannyThreshold);
			createTrackbar(accumulatorThresholdTrackbarName, windowName, &accumulatorThreshold, maxAccumulatorThreshold);
			createTrackbar(inverseResolutionName, windowName, &inverseResolution, maxInverseResolution);

			cannyThreshold = std::max(cannyThreshold, 1);
			accumulatorThreshold = std::max(accumulatorThreshold, 1);

			HoughDetection(src_gray, src, cannyThreshold, accumulatorThreshold, inverseResolution);
			//cvShowImage("V", frame);
			m.clear();
		}
		else{
			m[whites] = frameGray;
			cout << "This is frame number " << framecounter << endl;
			//cvShowImage("Current", frame);
		}
		
		// get user key
		char key = cvWaitKey(1);
		if (key == 27) break;
	}//end of while
	char k = cvWaitKey(0);
	return 0;
}