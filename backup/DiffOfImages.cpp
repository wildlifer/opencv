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
	
	IplImage* img1  = cvLoadImage("images/highway/shot0035.png");
	IplImage* img1Gray = cvCreateImage(cvGetSize(img1), IPL_DEPTH_8U, 1);
	cvCvtColor(img1,img1Gray,CV_BGR2GRAY);
	int whitesO = cvCountNonZero(img1Gray);
	cout << "No. of whites in original image are " << whitesO << "\n";

	IplImage* img2 = cvLoadImage("images/highway/shot0029.png");
	IplImage* img2Gray = cvCreateImage(cvGetSize(img2), IPL_DEPTH_8U, 1);
	cvCvtColor(img2, img2Gray, CV_BGR2GRAY);

	cvNamedWindow("Img Difference", CV_WINDOW_AUTOSIZE);
	IplImage* img3=cvCreateImage(cvGetSize(img2),IPL_DEPTH_8U,1);
	cvAbsDiff(img1Gray, img2Gray, img3);
	cvShowImage("Img Difference", img3);
	int whites = cvCountNonZero(img3);
	cout << "No. of whites are " << whites;
	cvWaitKey(0);
	return 0;
}