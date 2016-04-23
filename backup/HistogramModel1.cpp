#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

ofstream outFile;
ofstream outFile2;
ofstream outFile3;
ofstream outFile4;

double chiSqThreshold = 2;

void AllocateImages(IplImage* I){

}
int ObjectDetected(){
	return 1;
}

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
	const int cannyThresholdInitialValue = 100;
	const int accumulatorThresholdInitialValue = 20;
	const int maxAccumulatorThreshold = 400;
	const int maxCannyThreshold = 255;
	const int inverseResolutionInitialValue = 1;
	const int maxInverseResolution = 4;
	//Mat to display the type of vehicle


	void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold, int inverseResolution)
	{
		Mat vehicle = Mat::zeros(200, 1260, CV_8UC3);
		Mat clearedVehicle = Mat::zeros(200, 1260, CV_8UC3);
		// will hold the results of the detection
		std::vector<Vec3f> circles;
		// runs the actual detection
		HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 8, cannyThreshold, accumulatorThreshold, 14, 50);

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
		imshow("Original Video", display);
		cvMoveWindow("Original Video", 0, 0);
		//cv::moveWindow("Original Video", src_display.cols, 0);
		imshow("Detected Vehicle", vehicle);
		cvMoveWindow("Detected Vehicle", 0, 450);
		//cv::moveWindow("Detected Vehicle", 0, src_display.rows + 30);
		cvWaitKey(1000);
		putText(clearedVehicle, "------", Point(vehicle.cols / 3 - vehicle.cols / 10, vehicle.rows / 2),
			FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
		imshow("Detected Vehicle", clearedVehicle);
		cvMoveWindow("Detected Vehicle", 0, 450);
	}
}
int main(int argc, char** argv)
{
	VideoCapture cap("videos/1.5.mp4");
	cvNamedWindow("Original Video", 0);
	cvResizeWindow("Original Video", 600, 400);

	Mat background, hsv_background;
	Mat nextframe, hsv_nextframe;
	cap >> background;//discarding first frame
	cap >> background;
	cvtColor(background, hsv_background, COLOR_BGR2HSV);

	imshow("Original Video", hsv_background);
	cvMoveWindow("Original Video", 0, 0);

	/// Using 50 bins for hue and 60 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };

	outFile.open("data/2.1.txt");
	outFile2.open("data/2.2.txt");
	outFile3.open("data/2.3.txt");
	outFile4.open("data/2.4.txt");

	/// Histograms
	MatND hist_background;
	MatND hist_nextframe;
	//first frame
	calcHist(&hsv_background, 1, channels, Mat(), hist_background, 2, histSize, ranges, true, false);
	normalize(hist_background, hist_background, 0, 1, NORM_MINMAX, -1, Mat());

	//second frame
	cap >> nextframe;
	cvtColor(nextframe, hsv_nextframe, COLOR_BGR2HSV);
	calcHist(&hsv_nextframe, 1, channels, Mat(), hist_nextframe, 2, histSize, ranges, true, false);
	normalize(hist_nextframe, hist_nextframe, 0, 1, NORM_MINMAX, -1, Mat());
	
	int compare_method = 1;
	double initChange = compareHist(hist_background, hist_nextframe, compare_method);
	cout << "Init change is " << initChange <<endl;
	double prevChange = 0.0;
	bool buffer = false;
	double meanChange = 0.0;
	int changedFrames = 0;
	int pause = 0;
	int prev = 0;
	while (1){
		cap >> nextframe;
		if (nextframe.empty()) break;
		cvtColor(nextframe, hsv_nextframe, COLOR_BGR2HSV);
		calcHist(&hsv_nextframe, 1, channels, Mat(), hist_nextframe, 2, histSize, ranges, true, false);
		normalize(hist_nextframe, hist_nextframe, 0, 1, NORM_MINMAX, -1, Mat());
		imshow("Original Video", nextframe);
		cvMoveWindow("Original Video", 0, 0);
		double change = compareHist(hist_background, hist_nextframe, compare_method);	
		cout << "abs(change - initChange) ->>> " << abs(change - initChange) << endl;
		if (abs(change - initChange)<0.5){
			buffer = false;
			changedFrames = 1;
			meanChange = 0.0;
		}
		if (change > chiSqThreshold && buffer==false){
			changedFrames++;
			meanChange += change;
			cout << "Mean is: " << meanChange/changedFrames << endl;
			cout << "****************************" << endl;
			cout << "Change Detected"<<endl;
			if (prevChange > change && prev==0){
				cout << "Sub Best Frame Detected" << endl;
				//buffer = true;
				prev++;
				cvWaitKey(0);
			}
			else if (prevChange > change && prev == 1 && abs(prevChange-change)>meanChange){
				cout << "Best Frame Detected" << endl;
				buffer = true;
				prev==0;
				cvWaitKey(0);
			}
			else if (prevChange < change && prev == 1){
				prev = 0;
			}
		}
		printf("Change is [%f] ",change );
		printf("Prev Change [%f] ", prevChange);
		printf("prev [%d] ", prev);
		printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
		prevChange = change;
		pause = cvWaitKey(10);
		if (pause == 32){
			pause = cvWaitKey(0);
		}
	}
	outFile.close();
	outFile2.close();
	outFile3.close();
	outFile4.close();
	int a = cvWaitKey(0);
	return 0;
}