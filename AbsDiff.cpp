#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

double MIN_VEHICLE_AREA = 400;

long whites = 0;
void AllocateImages(IplImage* I){
	
	
}
int main(int argc, char** argv)
{
	CvCapture* capture = cvCreateFileCapture("videos/4.1.mp4");
	long initWhites = 0;
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
	/*IplImage* secondFrame = cvQueryFrame(capture);
	IplImage* thirdFrame = cvQueryFrame(capture);
	
	IplImage*  frame2 = cvCreateImage(cvSize(firstFrame->width / 3.5, firstFrame->height / 3), IPL_DEPTH_8U, 3);
	IplImage*  frame3 = cvCreateImage(cvSize(firstFrame->width / 3.5, firstFrame->height / 3), IPL_DEPTH_8U, 3);
	AllocateImages(frame);
	//get the diff of first and second frame
	cvResize(firstFrame, frame, 3);
	cvResize(secondFrame, frame2, 3);
	cvResize(thirdFrame, frame3, 3);
	cvAbsDiff(frame, frame2, diffFrame);
	cvCvtColor(diffFrame, frameGray, CV_BGR2GRAY);
	//get the whites of first frame difference
	initWhites = cvCountNonZero(frameGray);
	cout << "Init Whites  are " << (initWhites) << endl;
	//
	cvCopy(frame, tempFrame);
	cvAbsDiff(tempFrame, frame3, diffFrame);
	cvCvtColor(diffFrame, frameGray, CV_BGR2GRAY);
	//get the whites of first frame difference
	
	cout << "Init Whites  1 are " << (initWhites) << endl;
	cvCopy(frame, tempFrame,NULL);
	//tempFrame = frame;
	//cvShowImage("tempFrame", tempFrame);
	//cvMoveWindow("tempFrame", (firstFrame->width / 3) + 10, (firstFrame->height / 3) + 10);
	*/
	int b = 0;
	bool first = true;
	long initwhites = 0;
	long whites = 0;
	std::vector<KeyPoint> keypoints;
	while (1) {
		newFrame = cvQueryFrame(capture);
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
		cvAbsDiff(oldFrame,frameGray, diffFrame);
		cvAbsDiff(oldFrame,frameGray, diffFrame);
		
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
			if (area > 300){
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
		//imshow("original after rect", original);

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