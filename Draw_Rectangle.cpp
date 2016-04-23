#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
	int a = cvWaitKey(100000);
	vector<vector<Point> > v;

	Mat originalimage;
	Mat image;

	// Loads the original image
	originalimage = imread("images/car11.png");
	CvCapture* capture = cvCreateFileCapture("videos/2.mp4");
	IplImage* newFrame = cvQueryFrame(capture);
	IplImage*  frame = cvCreateImage(cvSize(newFrame->width / 3.5, newFrame->height / 3), IPL_DEPTH_8U, 3);
	// Converts original image to an 8UC1 image
	cvtColor(originalimage, image, CV_BGR2GRAY);

	// Finds contours
	findContours(image, v, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	// Finds the contour with the largest area
	vector<Rect> boundRect(v.size());
	Point pt1, pt2;
	cout << "No of contours found: " << v.size() << endl;
	for (int i = 0; i < v.size(); i++)
	{
		//cout << "rect no " << boundRect[i] << endl;
		double a = contourArea(v[i], false);
		cout << "Area of contour no " << i << ": " << a << endl;
		boundRect[i] = boundingRect(Mat(v[i]));
		cout << "rect no " << i << ": "<< boundRect[i] << endl;

		pt1.x = boundRect[i].x;
		pt1.y = boundRect[i].y;
		pt2.x = boundRect[i].x + boundRect[i].width;
		pt2.y = boundRect[i].y + boundRect[i].height;
		rectangle(image, pt1, pt2, Scalar(255), 1);
		cv::waitKey(30);
		
		fflush(stdin);
	}

	cvNamedWindow("Ejemplo", CV_WINDOW_AUTOSIZE);
	imshow("Ejemplo", image);
	cvWaitKey(100000);
	return 0;
}