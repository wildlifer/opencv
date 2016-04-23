#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
IplImage* background;
ofstream outFile;

long threshouldWhiteDifference = 3000;
long upperThreshouldWhiteDifference = 10000;
long upperThreshouldWhiteDifferenceIncrement = 5000;
long threshouldGrayWhiteDifference = 60;
long noisyWhitesThreshould = 2000;
long diffWhitesThreshould = 500;
long grayNoisyWhitesThreshould = 5;
void AllocateImages(IplImage* I){


}
int ObjectDetected(){

	return 1;
}
bool Checkbuffer(long initW, long currentW){
	if ((initW - currentW) < noisyWhitesThreshould)
		return false;
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
	const int cannyThresholdInitialValue = 105;
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
		HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 14, cannyThreshold,
			accumulatorThreshold, 18, 35);

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
		case 0:outFile << " No Circles " << circles.size() << endl;
			putText(vehicle, "-----------", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
			break;
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
		case 5:outFile << " Truck1 - " << circles.size() << endl;
			putText(vehicle, "Truck1", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
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
	Mat src_base, hsv_base;
	Mat src_test1, hsv_test1;
	Mat src_test2, hsv_test2;
	Mat src_test3, hsv_test3;
	Mat src_test4, hsv_test4;
	Mat hsv_half_down;

	src_base = imread("images/histogram/shot0046.png", 1);
	src_test1 = imread("images/histogram/shot0002.png", 1);
	src_test2 = imread("images/histogram/shot0003.png", 1);
	src_test3 = imread("images/histogram/shot0004.png", 1);
	src_test4 = imread("images/histogram/shot0044.png", 1);

	/// Convert to HSV
	cvtColor(src_base, hsv_base, COLOR_BGR2HSV);
	cvtColor(src_test1, hsv_test1, COLOR_BGR2HSV);
	cvtColor(src_test2, hsv_test2, COLOR_BGR2HSV);
	cvtColor(src_test3, hsv_test3, COLOR_BGR2HSV);
	cvtColor(src_test4, hsv_test4, COLOR_BGR2HSV);

	hsv_half_down = hsv_base(Range(hsv_base.rows / 2, hsv_base.rows - 1), Range(0, hsv_base.cols - 1));

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
	MatND hist_base;
	MatND hist_half_down;
	MatND hist_test1;
	MatND hist_test2;
	MatND hist_test3;
	MatND hist_test4;

	/// Calculate the histograms for the HSV images
	calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_half_down, 1, channels, Mat(), hist_half_down, 2, histSize, ranges, true, false);
	normalize(hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
	normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_test2, 1, channels, Mat(), hist_test2, 2, histSize, ranges, true, false);
	normalize(hist_test2, hist_test2, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_test3, 1, channels, Mat(), hist_test3, 2, histSize, ranges, true, false);
	normalize(hist_test3, hist_test3, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_test4, 1, channels, Mat(), hist_test4, 2, histSize, ranges, true, false);
	normalize(hist_test4, hist_test4, 0, 1, NORM_MINMAX, -1, Mat());

	/// Apply the histogram comparison methods
	for (int i = 0; i < 4; i++)
	{
		int compare_method = i;
		double base_base = compareHist(hist_base, hist_base, compare_method);
		//double base_half = compareHist(hist_base, hist_half_down, compare_method);
		double base_test1 = compareHist(hist_base, hist_test1, compare_method);
		double base_test2 = compareHist(hist_base, hist_test2, compare_method);
		double base_test3 = compareHist(hist_base, hist_test3, compare_method);
		double base_test4 = compareHist(hist_base, hist_test4, compare_method);

		printf(" Method [%d]  Perfect,  %f \n", i, base_base);
		//printf(" Method [%d]  Base-Half %f \n", i, base_half);
		printf(" Method [%d]  Base-Test(1) %f \n", i, base_test1);
		printf(" Method [%d]  Base-Test(2) %f \n", i, base_test2);
		printf(" Method [%d]  Base-Test(3) %f \n", i, base_test3);
		printf(" Method [%d]  Base-Test(3) %f \n", i, base_test4);
		printf("--------------------------------------------\n");
		
	}

	printf("Done \n");
	cvWaitKey(10000);
	imshow("Image",src_base);
	cvWaitKey(0);
	return 0;
}