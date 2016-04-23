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
ofstream outFile2;
ofstream outFile3;
ofstream outFile4;

double BHATTATHRES = 0.15;
double BHATTAMINTHRES = 0.058;
double BHATTA_BEST_FRAME = 0.00;
double BHATTA_BEST_FRAME_DIFF = 0.03;

double CHISQTHRES = 2;
double CHISQMINTHRES = 0.6;
double CHISQ_BEST_FRAME = 0.00;
double CHISQ_BEST_FRAME_DIFF = 1;

const int LEARN_FRAME_LIMIT = 8;
int INTERSEC = 2;

Mat src; Mat src_gray; Mat src_gray1;
int thresh = 120;
int max_thresh = 255;
RNG rng(12345);

/** @function thresh_callback */
void thresh_callback(int, Mat src_gray)
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	cout << "*********************************************" << endl;
	/// Approximate contours to polygons + get bounding rects and circles
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>center(contours.size());
	vector<float>radius(contours.size());
	double* height = new double[contours.size()];
	double* width = new double[contours.size()];
	for (int i = 0; i < contours.size(); i++)
	{
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		cout << "rectangle hight is" << boundRect[i].height << endl;
		cout << "rectangle width is" << boundRect[i].width << endl;
		height[i] = boundRect[i].height;
		width[i] = boundRect[i].width;
		minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}


	/// Draw polygonal contour + bonding rects + circles
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
	}

	/// Show in a window
	namedWindow("Contours", 0);
	cvResizeWindow("Contours", 600, 400);
	imshow("Contours", drawing);
	//cvMoveWindow("Contours", 600, 0);
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
	int THRESHOLD = 22;
	//Mat to display the type of vehicle


	void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold, int inverseResolution)
	{
		Mat vehicle = Mat::zeros(200, 1260, CV_8UC3);
		Mat clearedVehicle = Mat::zeros(200, 1260, CV_8UC3);
		// will hold the results of the detection
		std::vector<Vec3f> circles;
		// runs the actual detection
		//HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 8, cannyThreshold, accumulatorThreshold, 14, 40);
		HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 14, cannyThreshold, accumulatorThreshold, 14, 35);
		const int noOfCircles = circles.size();
		int noOfFinalCircles = 0;
		int **adjacency = new int*[noOfCircles];
		for (int i = 0; i < noOfCircles; i++)
			adjacency[i] = new int[noOfCircles];
		//int *rows = new int[noOfCircles*noOfCircles];
		//Discard the falsely detected circles
		int* adjacencySum = new int[noOfCircles];
		for (int i = 0; i < noOfCircles; i++){
			adjacencySum[i] = 0;
		}
		for (int i = 0; i < noOfCircles; i++){
			//cout << "Circle No " << i << " with center "<<
			for (int j = 0; j < noOfCircles; j++){
				if (i == j){
					adjacency[i][j] = 1;
				}
				else if (abs(cvRound(circles[i][1]) - cvRound(circles[j][1])) < THRESHOLD) {
					adjacency[i][j] = 1;
				}
				else{
					adjacency[i][j] = 0;
				}
				cout  << adjacency[i][j] << ",";
			}
			cout << endl;
		}
		//sum the adjacency matrix
		//vector<int> sum = new vector[4];
		for (int i = 0; i< noOfCircles; i++){
			for (int j = 0; j < noOfCircles; j++){
				adjacencySum[i] += adjacency[i][j];
			}
		}
		double a[3] = {1,2,3};
		//src_gray1 = src_gray(Rect(0, src_gray.rows / 3, src_gray.cols, src_gray.rows / 3));
		//thresh_callback(0, src_gray1);
		//cvWaitKey(0);
		// clone the colour, input image for displaying purposes
		Mat display = src_display.clone();
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			cout << "Adjacency sum is " << adjacencySum[i] << " with y coordinate as " << cvRound(circles[i][1]) << " with radius "<<circles[i][2] <<endl;
			if (adjacencySum[i] < *std::max_element(adjacencySum, adjacencySum + noOfCircles)){
				continue;
			}
			// circle center
			circle(display, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			circle(display, center, radius, Scalar(0, 0, 255), 3, 8, 0);
			noOfFinalCircles++;
		}
		switch (noOfFinalCircles){
		case 0:outFile << " False - " << circles.size() << endl;
			putText(clearedVehicle, "------", Point(vehicle.cols / 3 - vehicle.cols / 10, vehicle.rows / 2),
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

void DetectCircles(Mat src){
	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
	cvNamedWindow("Blurred Image", 0);
	cvResizeWindow("Blurred Image", 600, 400);
	imshow("Blurred Image",src_gray);
	cvMoveWindow("Blurred Image", 600, 0);
	//declare and initialize both parameters that are subject to change
	int cannyThreshold = cannyThresholdInitialValue;
	int accumulatorThreshold = accumulatorThresholdInitialValue;
	int inverseResolution = inverseResolutionInitialValue;
	cannyThreshold = std::max(cannyThreshold, 1);
	accumulatorThreshold = std::max(accumulatorThreshold, 1);
	HoughDetection(src_gray, src, cannyThreshold, accumulatorThreshold, inverseResolution);
}
int main(int argc, char** argv)
{
	VideoCapture cap("videos/3.1.mp4");
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
	MatND store[LEARN_FRAME_LIMIT];

	//first frame
	calcHist(&hsv_background, 1, channels, Mat(), hist_background, 2, histSize, ranges, true, false);
	normalize(hist_background, hist_background, 0, 1, NORM_MINMAX, -1, Mat());

	//second frame
	cap >> nextframe;
	cvtColor(nextframe, hsv_nextframe, COLOR_BGR2HSV);
	calcHist(&hsv_nextframe, 1, channels, Mat(), hist_nextframe, 2, histSize, ranges, true, false);
	normalize(hist_nextframe, hist_nextframe, 0, 1, NORM_MINMAX, -1, Mat());

	int CHISQUARE = 1;
	int BHATTA = 3;
	double initChangeCHI = compareHist(hist_background, hist_nextframe, CHISQUARE);
	double initChangeBHATTA = compareHist(hist_background, hist_nextframe, BHATTA);
	cout << "Init change CHI is " << initChangeCHI << endl;
	cout << "Init change BHATTA is " << initChangeBHATTA << endl;
	//cvWaitKey(0);
	double prevChangeCHI = 0.0;
	double prevChangeBHATTA = 0.0;
	double prevChangeINTERSEC = 0.0;
	int potentialBackground = 0;
	bool buffer = false;
	int bufferCount = 0;
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



		double changeCHI = compareHist(hist_background, hist_nextframe, CHISQUARE);
		double changeBHATTA = compareHist(hist_background, hist_nextframe, BHATTA);
		double changeINTERSEC = compareHist(hist_background, hist_nextframe, INTERSEC);

		printf("Change in CHISQ is [%f] ", changeCHI);
		printf("Change in BHATTA is [%f] ", changeBHATTA);
		printf("Change in INTERSEC is [%f] ", changeINTERSEC);
		cout<< endl;
		if (buffer == true)
			bufferCount++;
		if (bufferCount == 45){
			buffer = false;
			cout << "Toggling buffer to false" << endl;
			//cvWaitKey(0);
			bufferCount = 0;
		}
		//if (abs(changeCHI - initChangeCHI)<CHISQMINTHRES){// || abs(changeBHATTA - initChangeBHATTA)<BHATTAMINTHRES){
		if (abs(changeBHATTA)<BHATTAMINTHRES){
			buffer = false;
			bufferCount = 0;
			cout << "Continuing --------------------------------" << endl;
			CHISQ_BEST_FRAME = 0.0;
			potentialBackground = 0;
			continue;
		}
		//Learn new background
		//abs(changeBHATTA - BHATTAMINTHRES)<0.05
		if (abs(changeBHATTA)>BHATTAMINTHRES  && changeCHI>0.9 && changeCHI<1.2 &&
			CHISQ_BEST_FRAME - changeBHATTA>1 && BHATTA_BEST_FRAME - changeBHATTA>0.02){
			if (potentialBackground < LEARN_FRAME_LIMIT){
				store[potentialBackground] = hist_nextframe.clone();
				potentialBackground++;
				cout << "Learning " << endl;
				//cvWaitKey(0);
				continue;
			}
			else{
				hist_background = store[0];
				cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"<<endl;
				cout << "Changing Background with abs(changeBHATTA - BHATTAMINTHRES) = " << abs(changeBHATTA - BHATTAMINTHRES) << endl;
				cout << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" << endl;
				//cvWaitKey(0);
				potentialBackground = 0;
			}
		}
		cout << "Buffer is " << buffer << endl;
		//cout << "abs(change - initChange) ->>> " << abs(changeCHI - initChangeCHI) << " with buffer "<<buffer<< endl;
		if (changeCHI<300){
			if (changeCHI > CHISQTHRES && buffer == false){
				potentialBackground = 0;
				cout << "****************************" << endl;
				cout << "Change Detected" << endl;
				if (prevChangeCHI > changeCHI){// && abs(changeCHI - CHISQ_BEST_FRAME)>CHISQ_BEST_FRAME_DIFF){//Model 1
				//if (prevChangeBHATTA > changeBHATTA){//Model 2
					cout << "Prev is " << prevChangeCHI << " and current is "<< changeCHI<<endl;
					cout << "*********************************************" << endl;
					cout << "Best Frame Detected Using CHISQ " << endl;
					cout << "*********************************************" << endl;
					DetectCircles(nextframe);
					buffer = true;
					CHISQ_BEST_FRAME = changeCHI;
					BHATTA_BEST_FRAME = changeBHATTA;
					cvWaitKey(0);
				}
				/*else if (prevChange > change && prev == 1){
				cout << "Best Frame Detected" << endl;
				buffer = true;
				prev == 0;
				cvWaitKey(0);
				}
				else if (prevChange < change && prev == 1){
				prev = 0;
				}*/
			}
		}
		else{
			potentialBackground = 0;
			if (changeBHATTA > BHATTATHRES && buffer == false){
				cout << "****************************" << endl;
				cout << "Change Detected" << endl;
				if (prevChangeBHATTA > changeBHATTA){
					cout << "Best Frame Detected Using BHATTA" << endl;
					DetectCircles(nextframe);
					buffer = true;
					cvWaitKey(0);
				}

			}
		}

		//printf("Prev Change [%f] ", prevChange);
		cout << endl;
		printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
		//prevChange = changeCHI;
		prevChangeCHI = changeCHI;
		prevChangeBHATTA = changeBHATTA;
		prevChangeINTERSEC = changeINTERSEC;
		pause = cvWaitKey(1);
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