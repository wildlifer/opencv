#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

long whites = 0;
void AllocateImages(IplImage* I){


}
int main(int argc, char** argv)
{
	// Read image
	IplImage* im = cvLoadImage("coins.jpg", IMREAD_GRAYSCALE);
	namedWindow("Display window", WINDOW_AUTOSIZE);
	cvShowImage("Display window",im);

	CvCapture* capture = cvCreateFileCapture("videos/2.mp4");
	// Set up the detector with default parameters.
	SimpleBlobDetector::Params params;
	params.filterByColor = true;
	params.blobColor = 255;
	SimpleBlobDetector detector(params);

	// Detect blobs.
	std::vector<KeyPoint> keypoints;
	detector.detect(im, keypoints);

	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
	Mat im_with_keypoints;
	Mat m(im);
	drawKeypoints(m, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Show blobs
	//imshow("keypoints", im_with_keypoints);
	waitKey(0);
	int a = cvWaitKey(100000);
	
	return 0;
}