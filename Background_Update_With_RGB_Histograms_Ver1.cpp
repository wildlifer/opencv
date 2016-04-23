#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace cv;

ofstream outFile;
bool ROI_ENABLED = false;
double MIN_VEHICLE_AREA = 400;


long whites = 0;
const int INITIAL_LEARN_FRAME_LIMIT = 35;
const int LEARN_FRAME_LIMIT = 3;
int frameLearnCounter = 0;
int CHISQUARE = 1;
int CHISQUARE_THRES = 0;
int BHATTA = 3;
int BHATTA_THRES = 3;
int INTERSEC = 2;
int INTERSEC_THRES = 7;

int ORIGIN = 0;
int SHIFT = 350;
int WINDISTANCE = 20;

//crop parameters
double cropY = 0.30;
double cropHeight = 0.50;

//window dispay parameters
int max_rows = 300;
int max_cols = 400;

//color weights
double R_WEIGHT = 0.40;
double G_WEIGHT = 0.26;
double B_WEIGHT = 0.33;
int getBackgroundThreshold(vector<Mat> cap, int limit){
	int thr;
	vector<int> v(limit, 0);
	cout << "1" << endl;
	Mat background(cap[0]), nextframe;
	cout << "2" << endl;
	vector<Mat> bgr_planes_curr_frame;
	vector<Mat> bgr_planes_background;
	/// Establish the number of bins
	int histSize = 256;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;

	//Start here
	int width = background.cols;
	int height = background.rows;
	//adjust window parameters
	if (width < max_cols)
		max_cols = width;
	if (height < max_rows)
		max_rows = height;

	int cropFrameY = height*cropY;
	int cropFrameHeight = height*cropHeight;
	Rect roi(0, cropFrameY, width, cropFrameHeight);
	int i = 0;
	bool first = true;

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat b_hist_back, g_hist_back, r_hist_back;
	Mat b_hist_curr, g_hist_curr, r_hist_curr;
	while (i < limit - 1) {
		cout << "4" << endl;
		//new frame
		Mat nextframe(cap[i]);
		cout << "5" << endl;
		if (nextframe.empty()) break;
		if (first){
			nextframe.copyTo(background);
			first = false;
			if (ROI_ENABLED)
				background = background(roi);
			split(background, bgr_planes_background);
			cvNamedWindow("Background Video", CV_WINDOW_NORMAL);
			cvResizeWindow("Background Video", max_cols, max_rows);
			cvShowImage("Background Video", new IplImage(background));
			cvMoveWindow("Background Video", 0, 0);
			//background frame histogram
			calcHist(&bgr_planes_background[0], 1, 0, Mat(), b_hist_back, 1, &histSize, &histRange, uniform, accumulate);
			calcHist(&bgr_planes_background[1], 1, 0, Mat(), g_hist_back, 1, &histSize, &histRange, uniform, accumulate);
			calcHist(&bgr_planes_background[2], 1, 0, Mat(), r_hist_back, 1, &histSize, &histRange, uniform, accumulate);

			/// Normalize the result to [ 0, histImage.rows ]
			normalize(b_hist_back, b_hist_back, 0, histImage.rows, NORM_MINMAX, -1, Mat());
			normalize(g_hist_back, g_hist_back, 0, histImage.rows, NORM_MINMAX, -1, Mat());
			normalize(r_hist_back, r_hist_back, 0, histImage.rows, NORM_MINMAX, -1, Mat());
			cout << "Continuing" << endl;
			continue;
		}
		//get region of interest if requested
		if (ROI_ENABLED)
			nextframe = nextframe(roi);
		split(nextframe, bgr_planes_curr_frame);

		cvNamedWindow("Training Video", CV_WINDOW_NORMAL);
		cvResizeWindow("Training Video", max_cols, max_rows);
		cvShowImage("Training Video", new IplImage(nextframe));
		cvMoveWindow("Training Video", ORIGIN + SHIFT, 0);
		/// Compute the histograms:
		calcHist(&bgr_planes_curr_frame[0], 1, 0, Mat(), b_hist_curr, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes_curr_frame[1], 1, 0, Mat(), g_hist_curr, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes_curr_frame[2], 1, 0, Mat(), r_hist_curr, 1, &histSize, &histRange, uniform, accumulate);
		/// Normalize the result to [ 0, histImage.rows ]
		normalize(b_hist_curr, b_hist_curr, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(g_hist_curr, g_hist_curr, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(r_hist_curr, r_hist_curr, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		//Model parameters
		double b_changeCHI = compareHist(b_hist_back, b_hist_curr, CHISQUARE);
		double b_changeBHATTA = compareHist(b_hist_back, b_hist_curr, BHATTA);
		double b_changeINTERSEC = compareHist(b_hist_back, b_hist_curr, INTERSEC);

		double g_changeCHI = compareHist(g_hist_back, g_hist_curr, CHISQUARE);
		double g_changeBHATTA = compareHist(g_hist_back, g_hist_curr, BHATTA);
		double g_changeINTERSEC = compareHist(g_hist_back, g_hist_curr, INTERSEC);

		double r_changeCHI = compareHist(r_hist_back, r_hist_curr, CHISQUARE);
		double r_changeBHATTA = compareHist(r_hist_back, r_hist_curr, BHATTA);
		double r_changeINTERSEC = compareHist(r_hist_back, r_hist_curr, INTERSEC);

		double changeCHI = B_WEIGHT*b_changeCHI + R_WEIGHT*r_changeCHI + G_WEIGHT*g_changeCHI;
		cout << "Change in  Chi is " << changeCHI << endl;
		//if (changeCHI < 1){
		//	cout << "CHI less than 1!! Continuing" << endl;
		//	continue;
		//}

		//Mat b_hist_back, g_hist_back, r_hist_back;/*This statement  saved me*/
		//if (changeCHI >0 && (changeCHI) < 4000){
		//	//cout << "Updating" << endl;
		//	//this can be optimized
		//	Mat r(r_hist_curr);
		//	Mat g(g_hist_curr);
		//	Mat b(b_hist_curr);
		//	r.copyTo(r_hist_back);
		//	g.copyTo(g_hist_back);
		//	b.copyTo(b_hist_back);
		//}
		v[i] = (int)changeCHI;
		i++;
		bgr_planes_curr_frame.clear();
		int xx = cvWaitKey(10);
	}
	//calculate mean and standard deviation
	int sum = std::accumulate(v.begin(), v.end(), 0.0);
	int mean = (int)sum / v.size();

	double E = 0;
	// Quick Question - Can vector::size() return 0?
	double inverse = 1.0 / static_cast<double>(v.size());
	for (unsigned int i = 0; i<v.size(); i++)
	{
		E += pow(static_cast<double>(v[i]) - mean, 2);
	}
	double stdev = sqrt(inverse * E);
	thr = mean + (int)stdev;
	cout << "***************************************" << endl;
	cout << " Threshold is: " << thr << endl;
	cout << "***************************************" << endl;
	int xx;
	cin >> xx;
	cin.clear();
	cin.ignore(INT_MAX, '\n');
	cvDestroyWindow("Training Video");
	cvDestroyWindow("Background Video");
	return thr;
}

int main(int argc, char** argv)
{
	VideoCapture cap("videos/4.1.mp4");
	Mat background, hsv_background, gray_background;
	cap >> hsv_background;
	vector<Mat> initFrames(INITIAL_LEARN_FRAME_LIMIT);
	for (int i = 0; i < INITIAL_LEARN_FRAME_LIMIT; i++){
		cap >> initFrames[i];
	}
	int threshold = getBackgroundThreshold(initFrames, initFrames.size());
	CHISQUARE_THRES = threshold;;
	Mat nextframe, hsv_nextframe, gray_nextframe;
	Mat diff_frame;
	outFile.open("data/VehicleCassifier/changeInMetric.txt", 'w');

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes_curr_frame;
	vector<Mat> bgr_planes_background;

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist_back, g_hist_back, r_hist_back;
	Mat b_hist_curr, g_hist_curr, r_hist_curr;
	//Canny parameters
	Mat canny_output;
	int thresh = 50;
	int max_thresh = 255;

	//Start here
	cap >> background;
	int width = background.cols;
	int height = background.rows;

	int cropFrameY = height*cropY;
	int cropFrameHeight = height*cropHeight;
	Rect roi(0, cropFrameY, width, cropFrameHeight);
	//get ROI if requested
	if (ROI_ENABLED)
		background = background(roi);
	split(background, bgr_planes_background);
	//background frame histogram
	calcHist(&bgr_planes_background[0], 1, 0, Mat(), b_hist_back, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes_background[1], 1, 0, Mat(), g_hist_back, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes_background[2], 1, 0, Mat(), r_hist_back, 1, &histSize, &histRange, uniform, accumulate);
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist_back, b_hist_back, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist_back, g_hist_back, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist_back, r_hist_back, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	//Create window for original video display
	cvNamedWindow("Original Video", CV_WINDOW_NORMAL);
	cvResizeWindow("Original Video", max_cols, max_rows);
	cvMoveWindow("Original Video", ORIGIN + SHIFT, 0);

	//Create temporary images
	CvSize sz = cvGetSize(new IplImage(background));
	IplImage* oldFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);;
	IplImage* frame = cvCreateImage(sz, IPL_DEPTH_8U, 3);;
	IplImage* diffFrameBinary = cvCreateImage(sz, IPL_DEPTH_8U, 3);;

	IplImage* agray = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	IplImage* bgray = cvCreateImage(sz, IPL_DEPTH_8U, 1);;

	IplImage* abin = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	IplImage* bbin = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	IplImage* diff = cvCreateImage(sz, IPL_DEPTH_8U, 1);;

	cvNamedWindow("Diff Frame", 0);
	int ii = 0;
	//store previous images
	vector<Mat> matStore(LEARN_FRAME_LIMIT + 1);
	while (1) {
		//new frame
		cap >> nextframe;
		if (nextframe.empty()) break;
		//get region of interest
		if (ROI_ENABLED)
			nextframe = nextframe(roi);
		split(nextframe, bgr_planes_curr_frame);
		/// Compute the histograms:
		calcHist(&bgr_planes_curr_frame[0], 1, 0, Mat(), b_hist_curr, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes_curr_frame[1], 1, 0, Mat(), g_hist_curr, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes_curr_frame[2], 1, 0, Mat(), r_hist_curr, 1, &histSize, &histRange, uniform, accumulate);
		/// Normalize the result to [ 0, histImage.rows ]
		normalize(b_hist_curr, b_hist_curr, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(g_hist_curr, g_hist_curr, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(r_hist_curr, r_hist_curr, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		//Show Original Video  - IplImage model
		frame = new IplImage(nextframe);
		cvCvtColor(frame, bgray, CV_BGR2GRAY);

		cvNamedWindow("Original Video", CV_WINDOW_NORMAL);
		cvResizeWindow("Original Video", max_cols, max_rows);
		cvShowImage("Original Video", frame);
		cvMoveWindow("Original Video", ORIGIN + SHIFT, 0);

		//Show current background
		oldFrame = new IplImage(background);
		cvCvtColor(oldFrame, agray, CV_BGR2GRAY);

		cvNamedWindow("Current background", CV_WINDOW_NORMAL);
		cvResizeWindow("Current background", max_cols, max_rows);
		cvShowImage("Current background", oldFrame);
		cvMoveWindow("Current background", ORIGIN + SHIFT + WINDISTANCE + max_cols, 0);

		//Get the difference between background and current frame
		cvAbsDiff(agray, bgray, diff);
		cvSmooth(diff, diff, CV_BLUR);
		cvThreshold(diff, diff, 25, 255, CV_THRESH_BINARY);
		//cvAbsDiff(abin, bbin, diff);
		//absdiff(gray_background, gray_nextframe, diff_frame);

		//Mat Model
		/*absdiff(background, nextframe, diff_frame);*/
		cvNamedWindow("Diff Frame", CV_WINDOW_NORMAL);
		cvResizeWindow("Diff Frame", max_cols, max_rows);
		cvShowImage("Diff Frame", diff);
		cvMoveWindow("Diff Frame", ORIGIN + SHIFT + WINDISTANCE + max_cols, 350);

		//Model parameters
		double b_changeCHI = compareHist(b_hist_back, b_hist_curr, CHISQUARE);
		double b_changeBHATTA = compareHist(b_hist_back, b_hist_curr, BHATTA);
		double b_changeINTERSEC = compareHist(b_hist_back, b_hist_curr, INTERSEC);

		double g_changeCHI = compareHist(g_hist_back, g_hist_curr, CHISQUARE);
		double g_changeBHATTA = compareHist(g_hist_back, g_hist_curr, BHATTA);
		double g_changeINTERSEC = compareHist(g_hist_back, g_hist_curr, INTERSEC);

		double r_changeCHI = compareHist(r_hist_back, r_hist_curr, CHISQUARE);
		double r_changeBHATTA = compareHist(r_hist_back, r_hist_curr, BHATTA);
		double r_changeINTERSEC = compareHist(r_hist_back, r_hist_curr, INTERSEC);

		//Commenting green channel as it is causing huge discrepencies in the change
		double changeCHI = B_WEIGHT*b_changeCHI + R_WEIGHT*r_changeCHI + G_WEIGHT*g_changeCHI;
		double changeBHATTA = R_WEIGHT*r_changeBHATTA + B_WEIGHT*b_changeBHATTA + G_WEIGHT*g_changeBHATTA;

		cout << "Change in Chi is " << (changeCHI) << endl;

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		if (changeCHI < 0){
			//Draw the histogram of current frame
			for (int i = 1; i < histSize; i++)
			{
				line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist_curr.at<float>(i - 1))),
					Point(bin_w*(i), hist_h - cvRound(b_hist_curr.at<float>(i))),
					Scalar(255, 0, 0), 2, 8, 0);
			}
			cvNamedWindow("Histogram", CV_WINDOW_NORMAL);
			cvResizeWindow("Histogram", max_cols, max_rows);
			cvShowImage("Histogram", new IplImage(histImage));
			cvMoveWindow("Histogram", ORIGIN, 350);
			continue;
		}
		int i = 1;
		ii++;
		if (ii % 10 == 0){
			int xx;
			cin >> xx;
			cin.clear();
			cin.ignore(INT_MAX, '\n');
		}
		//Re-initialize the matrix. Very important
		Mat b_hist_back, g_hist_back, r_hist_back;
		////New Background
		if (abs(changeCHI) < CHISQUARE_THRES){
			frameLearnCounter++;
			cout << "Below Threshold with frame counter: " << frameLearnCounter << endl;
			//Keep previous copies of similar images
			//if (frameLearnCounter<LEARN_FRAME_LIMIT)
			//nextframe.copyTo(matStore[frameLearnCounter]);

			if (frameLearnCounter == LEARN_FRAME_LIMIT){
				frameLearnCounter = 0;
				//matStore.clear();
				//copy current frame to the background
				background = nextframe.clone();
				//nextframe.copyTo(background);
				r_hist_back = r_hist_curr.clone();
				g_hist_back = g_hist_curr.clone();
				b_hist_back = b_hist_curr.clone();
				bgr_planes_curr_frame.clear();
				cout << "Updating Background" << endl;
				/*int xx;
				cin >> xx;
				cin.clear();
				cin.ignore(INT_MAX, '\n');*/
			}

		}
		else if (abs(r_changeCHI)>CHISQUARE_THRES){
			frameLearnCounter = 0;
			cout << "Above Threshold with frame counter: " << frameLearnCounter << endl;
			//matStore.clear();
			//contours and blobs
			// Call this only when some vehicle is detected
			Mat src(diff);
			Mat original(bgray);
			vector<vector<Point> > contours;
			vector<Vec4i> hierarchy;
			//Canny(src, canny_output, thresh, thresh * 2, 3);

			//CV_RETR_EXTERNAL is important here. It helps to find only outer contours which is what we need
			findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
			vector<Rect> boundRect(contours.size());
			Point pt1, pt2;
			cout << "No of contours found: " << contours.size() << endl;
			double max_area = 0;
			for (int i = 0; i < contours.size(); i++){
				if (contourArea(contours[i], false)>max_area)
					max_area = contourArea(contours[i]);
			}
			for (int i = 0; i < contours.size(); i++)
			{
				double area = contourArea(contours[i], false);
				if (abs(area - max_area)< 1.0){
					//cout << "rect no " << boundRect[i] << endl;
					boundRect[i] = boundingRect(Mat(contours[i]));
					//cout << "rect no " << boundRect[i] << endl;
					pt1.x = boundRect[i].x;
					pt1.y = boundRect[i].y;
					pt2.x = boundRect[i].x + boundRect[i].width;
					pt2.y = boundRect[i].y + boundRect[i].height;
					rectangle(src, pt1, pt2, Scalar(255), 1);
					cvNamedWindow("Detected Vehicles", CV_WINDOW_NORMAL);
					cvResizeWindow("Detected Vehicles", max_cols, max_rows);
					imshow("Detected Vehicles", src);
					cvMoveWindow("Detected Vehicles", ORIGIN + SHIFT, 350);
					cv::waitKey(30);
				}
			}
		}

		outFile << r_changeCHI << endl;
		cvWaitKey(30);
	}
	int a = cvWaitKey(100000);
	return 0;
}