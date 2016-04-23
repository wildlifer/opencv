#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

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
	const int accumulatorThresholdInitialValue = 35;
	const int maxAccumulatorThreshold = 400;
	const int maxCannyThreshold = 255;
	const int inverseResolutionInitialValue=1;
	const int maxInverseResolution = 3;

	void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold, int inverseResolution)
	{
		// will hold the results of the detection
		std::vector<Vec3f> circles;
		// runs the actual detection
		HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 3, cannyThreshold, accumulatorThreshold, 10, 30);

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

		// shows the results
		imshow(windowName, display);
	}
}


int main(int argc, char** argv)
{
	ifstream file;
	file.open("images/CEC2015/Base/list1.txt",ios::in);
	string line;
	string imageList[20];
	int count = 0;
	while (std::getline(file, line)){
		cout << line << '\n';
		imageList[count] = line;
		count++;
	}

	//int a = waitKey(50000);
	for (int i = 0; i <= sizeof(imageList); i++){
		
		Mat src, src_gray;
		string path = "images/CEC2015/Base/" + imageList[i];
		src = imread(path, 1);

		/// Read the image
		/*if (i==1)
			src = imread("images/highway/truck1.png", 1);
		else if (i==2)
			src = imread("images/highway/truck2.png", 1);
		else
			src = imread("images/highway/truck3.png", 1);
		//src = imread(path, 1);*/

		if (!src.data)
		{
			std::cerr << "Invalid input image\n";
			std::cout << usage;
			return -1;
		}

		// Convert it to gray
		cvtColor(src, src_gray, COLOR_BGR2GRAY);

		// Reduce the noise so we avoid false circle detection
		GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

		//declare and initialize both parameters that are subjects to change
		int cannyThreshold = cannyThresholdInitialValue;
		int accumulatorThreshold = accumulatorThresholdInitialValue;
		int inverseResolution = inverseResolutionInitialValue;

		// create the main window, and attach the trackbars
		namedWindow(windowName, WINDOW_AUTOSIZE);
		createTrackbar(cannyThresholdTrackbarName, windowName, &cannyThreshold, maxCannyThreshold);
		createTrackbar(accumulatorThresholdTrackbarName, windowName, &accumulatorThreshold, maxAccumulatorThreshold);
		createTrackbar(inverseResolutionName, windowName, &inverseResolution, maxInverseResolution);


		int key = 0;
		while (key != 'q' && key != 'Q')
		{
			// those paramaters cannot be =0

			cannyThreshold = std::max(cannyThreshold, 1);
			accumulatorThreshold = std::max(accumulatorThreshold, 1);


			HoughDetection(src_gray, src, cannyThreshold, accumulatorThreshold, inverseResolution);

			// get user key
			key = waitKey(200);
		}
		//key = waitKey(50);
	}
	//int ch = 0;
	//while (ch != 'q' && ch != 'Q')
	//{
		//ch = waitKey(10);
	//}
	
	return 0;
}