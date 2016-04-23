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
	const int cannyThresholdInitialValue = 105;
	const int accumulatorThresholdInitialValue = 55;
	const int maxAccumulatorThreshold = 400;
	const int maxCannyThreshold = 155;
	const int inverseResolutionInitialValue = 1;
	const int maxInverseResolution = 2;

	void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold, int inverseResolution)
	{
		// will hold the results of the detection
		std::vector<Vec3f> circles;
		// runs the actual detection
		HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1.3, src_gray.rows / 2.3, cannyThreshold, accumulatorThreshold, 0, 80);

		/// Apply the Hough Transform to find the circles
		//HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1.98, src_gray.rows / 5, 250, 50, 0, 30);

		// clone the colour, input image for displaying purposes
		Mat display = src_display.clone();
		std::cout << "No of circles detected are: " << circles.size();
		std::cout << "\n";

		int b = waitKey(100);
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
	//int a = waitKey(50000);

	//CvCapture* capture = cvCreateFileCapture("videos/6.avi");
	CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
	IplImage* frame;
	while (1) {
		frame = cvQueryFrame(capture);
		if (!frame) break;
		//cvShowImage("Example2", frame);
		//char c = cvWaitKey(33);

		Mat src(frame), src_gray;
		//src=Mat::matCon(frame);
		//string path = "images/highway/" + imageList[i];
		//src = imread("images/highway/truck1.png", 1);


		if (!src.data)
		{
			std::cerr << "Invalid input frame\n";
			std::cout << usage;
			int b = waitKey(300);
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
		//while (key != 'q' && key != 'Q')
		//{
		// those paramaters cannot be =0

		cannyThreshold = std::max(cannyThreshold, 1);
		accumulatorThreshold = std::max(accumulatorThreshold, 1);


		HoughDetection(src_gray, src, cannyThreshold, accumulatorThreshold, inverseResolution);

		// get user key
		key = waitKey(10);
		if (key == 27) break;
	}//end of while
	//}
	//key = waitKey(50);



	return 0;
}