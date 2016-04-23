#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

/** @function main */
int main(int argc, char** argv)
{
	Mat src, src_gray;

	/// Read the image
	src = imread("images/highway/test.png", 1);

	if (!src.data)
	{
		int a;	cout << "Cant load image!!!";cin >> a;return -1;
	}

	/// Convert it to gray
	cvtColor(src, src_gray, CV_BGR2GRAY);
		
	//Show the gray scale image with Blur
	namedWindow("Gray scale with blur image demo", CV_WINDOW_AUTOSIZE);
	imshow("Gray scale with blur", src_gray);
	
	/// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);

	//Show the gray scale image
	namedWindow("Gray scale image demo", CV_WINDOW_AUTOSIZE);
	imshow("Gray scale", src_gray);

	vector<Vec3f> circles;

	/// Apply the Hough Transform to find the circles
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1.98, src_gray.rows/4, 250, 50, 1, 30);
	//HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, 5, 200, 140, 0, 0);

	/// Draw the circles detected
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// circle outline
		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}

	/// Show your results
	namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
	imshow("Hough Circle Transform Demo", src);

	waitKey(0);
	return 0;
}