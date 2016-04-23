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
	const std::string windowName = "Hough Circle Detection";
	const std::string detectedVehicle = "Detected Vehicle";
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
	//Mat to display the type of vehicle


	void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold, int inverseResolution)
	{
		Mat vehicle = Mat::zeros(200, 1300, CV_8UC3);
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
		case 1:outFile << " False - " << circles.size() << endl;
			putText(vehicle, "False", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
			break;
		case 2:outFile << " Car - " << circles.size() << endl;
			putText(vehicle, "Car", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
			break;
		case 3:outFile << " Car - " << circles.size() << endl;
			putText(vehicle, "Truck", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false); break;
		case 4:outFile << " Truck - " << circles.size() << endl;
			putText(vehicle, "Truck", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
			break;
		default: outFile << " SpaceShip - " << circles.size() << endl;
			putText(vehicle, "SpaceShip", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
			break;
		}

		// shows the results
		imshow(windowName, display);
		cv::moveWindow(windowName, src_display.cols, 0);
		imshow(detectedVehicle, vehicle);
		cv::moveWindow(detectedVehicle, 0, src_display.rows + 30);
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
		cout << it->first << endl;
		if (i == m.size() / 2){
			key = it->first;
			break;
		}
	}
	mean = (long)mean / m.size();
	cout << "Mean is " << mean << endl;
	cout << "Best found at " << key << " at location " << i << endl;
	//cvShowImage("Best1", m[key]);
	return  m[key];
}

int main(int argc, char** argv)
{
	CvCapture* capture = cvCreateFileCapture("videos/2.mp4");
	outFile.open("data\\Classifier.txt");
	cout << "In MAIN" << endl;
	IplImage* firstFrame = cvQueryFrame(capture);
	IplImage*  frame = cvCreateImage(cvSize(firstFrame->width / 3, firstFrame->height / 3), IPL_DEPTH_8U, 3);
	cvResize(firstFrame, frame, 3);

	cout << "after cvResize" << endl;
	cvNamedWindow("Original Video");
	IplImage* frameGray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	IplImage* bestFrame = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 3);
	cvCvtColor(frame, frameGray, CV_BGR2GRAY);
	long initialWhites = cvCountNonZero(frameGray);
	long whites = 0;
	long thresholdWhitecount = 11;
	int counter = 0;
	long framecounter = 0;
	std::map<long, IplImage*> m;
	
	cout << "Before While" << endl;
	while (1) {
		firstFrame = cvQueryFrame(capture);
		if (!firstFrame) break;
		cout << "Before Cv in  While" << endl;
		cvResize(firstFrame, frame, 3);
		cout << "After Cv in  While" << endl;
		cvShowImage("Original Video", frame);
		cvMoveWindow("Original Video", 0, 0);

		cvWaitKey(100);
		
		if (!frame) break;
		cvCvtColor(frame, frameGray, CV_BGR2GRAY);
		whites = cvCountNonZero(frameGray);
		framecounter++;
		
		if ((initialWhites - whites) < thresholdWhitecount) continue;
		
		if (++counter == 5){
			counter = 0;
			cout << "Frame Found" << endl;
			bestFrame = getBestFrame(m);
			Mat src(frame), src_gray(bestFrame);
			// Reduce the noise so we avoid false circle detection
			GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

			//declare and initialize both parameters that are subjects to change
			int cannyThreshold = cannyThresholdInitialValue;
			int accumulatorThreshold = accumulatorThresholdInitialValue;
			int inverseResolution = inverseResolutionInitialValue;

			// create the main window, and attach the trackbars
			namedWindow(windowName, WINDOW_AUTOSIZE);
			
			cannyThreshold = std::max(cannyThreshold, 1);
			accumulatorThreshold = std::max(accumulatorThreshold, 1);
			HoughDetection(src_gray, src, cannyThreshold, accumulatorThreshold, inverseResolution);
			m.clear();
		}
		else{
			m[whites] = frameGray;
			cout << "This is frame number " << framecounter << endl;
		}

		// get user key
		char key = cvWaitKey(33);
		if (key == 27) break;
	}//end of while
	cout << "Waiting" << endl;
	char k = cvWaitKey(0);
	return 0;
}