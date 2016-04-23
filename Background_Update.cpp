#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

long whites = 0;
const int LEARN_FRAME_LIMIT = 8;
int main(int argc, char** argv)
{
	CvCapture* capture = cvCreateFileCapture("videos/4.1.mp4");
	VideoCapture cap("videos/4.1.mp4");
	long initWhites = 0;
	Mat background, hsv_background;
	Mat nextframe, hsv_nextframe;
	Mat diff_frame;

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
	/// Histograms
	MatND hist_background;
	MatND hist_nextframe;
	MatND store[LEARN_FRAME_LIMIT];

	//first frame
	calcHist(&hsv_background, 1, channels, Mat(), hist_background, 2, histSize, ranges, true, false);
	normalize(hist_background, hist_background, 0, 1, NORM_MINMAX, -1, Mat());



	IplImage* newFrame = cvQueryFrame(capture);
	//IplImage*  frame = cvCreateImage(cvSize(newFrame->width / 3.5, newFrame->height / 3), IPL_DEPTH_8U, 3);
	IplImage*  frame = cvCreateImage(cvSize(newFrame->width, newFrame->height), IPL_DEPTH_8U, 3);
	IplImage* diffFrame;
	IplImage* oldFrame;
	IplImage* frameGray;
	CvSize sz = cvGetSize(frame);
	CvScalar cwhite;

	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;
	params.filterByColor = true;
	params.blobColor = 255;
	SimpleBlobDetector blob_detector(params);

	diffFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	oldFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	frameGray = cvCreateImage((sz), IPL_DEPTH_8U, 1);
	cvResize(newFrame, frame, 3);
	
	int b = 0;
	bool first = true;
	long initwhites = 0;
	long whites = 0;
	std::vector<KeyPoint> keypoints;
	while (1) {
		newFrame = cvQueryFrame(capture);
		cap >> nextframe;
		if (!newFrame) break;
		cvResize(newFrame, frame, 3);
		cvCvtColor(frame, frameGray, CV_BGR2GRAY);
		if (first){
			diffFrame = cvCloneImage(frameGray);
			oldFrame = cvCloneImage(frameGray);
			first = false;
			initWhites = cvCountNonZero(frameGray);
			cvConvertScale(frameGray, oldFrame, 1.0, 0.0);
			cout << "Init whites are " << (initWhites) << endl;
			continue;
		}
		whites = cvCountNonZero(frameGray);
		//Calculate diff
		cvAbsDiff(oldFrame, frameGray, diffFrame);

		//cout << "Current Frame whites are " << (whites) << endl;
		//cout << "Change in whites are " << (whites - initWhites) << endl;
		cvSmooth(diffFrame, diffFrame, CV_BLUR);
		cvThreshold(diffFrame, diffFrame, 25, 255, CV_THRESH_BINARY);

		//show images
		cvShowImage("Original Video", frame);
		cvMoveWindow("Original Video", 0, 0);
		//cvShowImage("Original Gray", frameGray);
		//cvMoveWindow("Original Gray", 100, 100);
		cvShowImage("Diff Frame", diffFrame);
		cvMoveWindow("Diff Frame", (newFrame->width / 3) + 10, 0);

		//contours and blobs
		Mat src;
		src = diffFrame;
		Mat original(frameGray);
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(src, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<Rect> boundRect(contours.size());
		Point pt1, pt2;
		cout << "No of contours found: " << contours.size() << endl;
		for (int i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i], false);
			if (area > 400){
				//cout << "rect no " << boundRect[i] << endl;
				boundRect[i] = boundingRect(Mat(contours[i]));
				//cout << "rect no " << boundRect[i] << endl;
				pt1.x = boundRect[i].x;
				pt1.y = boundRect[i].y;
				pt2.x = boundRect[i].x + boundRect[i].width;
				pt2.y = boundRect[i].y + boundRect[i].height;
				rectangle(src, pt1, pt2, Scalar(255), 1);
				imshow("Rectangles", src);
				cv::waitKey(30);
			}
			//
		}
		

		/*Mat im_with_keypoints;
		blob_detector.detect(diffFrame, keypoints);
		for (int i = 0; i<keypoints.size(); i++){
		float X = keypoints[i].pt.x;
		float Y = keypoints[i].pt.y;
		cout << "X = " << X << " and Y = "<< Y << endl;
		cout << "--------------------------" << endl;
		}
		drawKeypoints(src, keypoints, im_with_keypoints, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		// Show blobs
		imshow("keypoints", im_with_keypoints);
		*/

		//New Background
		cvConvertScale(frameGray, oldFrame, 1.0, 0.0);
		cvWaitKey(10);
	}
	int a = cvWaitKey(100000);
	cvReleaseCapture(&capture);
	return 0;
}