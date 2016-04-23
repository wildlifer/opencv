#include "helper_cv.h"
int frameLearnCounter_L = 0;
int frameLearnCounter_G = 0;
int CHISQUARE_THRES = 0;

int main(int argc, char** argv)
{
	vector<Mat> initFrames;
	ofstream outFile;
	VideoCapture cap("videos/4.5.mp4");
	int threshold = getBackgroundThreshold(cap, INITIAL_LEARN_FRAME_LIMIT);
	CHISQUARE_THRES = threshold;;
	Mat background, hsv_background, gray_background;
	Mat nextframe, hsv_nextframe, gray_nextframe;
	Mat diff_frame;
	outFile.open("data/VehicleCassifier/changeInMetric.txt", 'w');

	/// Separate the image in 3 places ( B, G and R )/
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

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	/// Normalize the result to [ 0, histImage.rows ]

	vector<Mat>hist = getNormalizedHist(background, histSize, range, uniform, accumulate);
	hist[0].copyTo(r_hist_back);
	hist[1].copyTo(g_hist_back);
	hist[2].copyTo(b_hist_back);
	//Create temporary images
	CvSize sz = cvGetSize(new IplImage(background));
	Mat agray(background);
	Mat bgray(background);
	Mat diff(background);

	IplImage* abin = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	IplImage* bbin = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	

	int ii = 0;
	//store previous images
	vector<Mat> matStore(LEARN_FRAME_LIMIT_LESS_THRES + 1);
	Mat rep_background; //replaceable backgrond
	Mat b_hist_back_rep, g_hist_back_rep, r_hist_back_rep;
	vector<int> hist_diff_store_less(LEARN_FRAME_LIMIT_LESS_THRES, 0);
	vector<int> hist_diff_store_greater(LEARN_FRAME_LIMIT_GREATER_THRES, 0);
	while (1) {
		//clear planes
		bgr_planes_curr_frame.clear();
		//new frame
		cap >> nextframe;
		if (nextframe.empty()) break;
		//get region of interest
		if (ROI_ENABLED)
			nextframe = nextframe(roi);
		vector<Mat>hist = getNormalizedHist(nextframe, histSize, range, uniform, accumulate);
		hist[0].copyTo(r_hist_curr);
		hist[1].copyTo(g_hist_curr);
		hist[2].copyTo(b_hist_curr);

		//Show Original Video  - IplImage model
		cvtColor(nextframe, bgray, CV_BGR2GRAY);
		medianBlur(bgray, bgray, APERTURE);
		createNMoveWindow("Original Video", max_cols, max_rows, nextframe, ORIGIN + SHIFT, 0);

		//Show current background
		cvtColor(background, agray, CV_BGR2GRAY);
		medianBlur(agray, agray, APERTURE);
		createNMoveWindow("Current background", max_cols, max_rows, background, ORIGIN + SHIFT + WINDISTANCE + max_cols, 0);

		//Get the difference between background and current frame
		absdiff(agray, bgray, diff);
		
		medianBlur(diff, diff, APERTURE);
		IplImage *I = new IplImage(diff);
		cvThreshold(I, I, 25, 255, CV_THRESH_BINARY);

		Mat m(I);
		diff = m.clone();
		
		createNMoveWindow("Diff", max_cols, max_rows, diff, ORIGIN + SHIFT + WINDISTANCE + max_cols+50, 350);
		//createNMoveWindow("Original", max_cols, max_rows, nextframe, ORIGIN + SHIFT + WINDISTANCE, 350);

		//Model parameters
		double b_changeCHI = compareHistograms(b_hist_back, b_hist_curr, CHISQUARE);
		double g_changeCHI = compareHistograms(g_hist_back, g_hist_curr, CHISQUARE);
		double r_changeCHI = compareHistograms(r_hist_back, r_hist_curr, CHISQUARE);
		double changeCHI   = B_WEIGHT*b_changeCHI + R_WEIGHT*r_changeCHI + G_WEIGHT*g_changeCHI;

		cout << "Change in Chi is " << (changeCHI) << endl;

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		int i = 1;

		Mat src=diff.clone();
		Mat original=bgray.clone();
		vector<vector<Point>> contours;
		Rect boundRec;
		vector<Vec4i> hierarchy;
		//Canny(src, canny_output, thresh, thresh * 2, 3);
		//CV_RETR_EXTERNAL is important here. It helps to find only outer contours which is what we need
		findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<Rect> boundRect(contours.size());
		Point pt1, pt2;
		drawContoursNDeNoise(diff, contours,20);
		//pause();
		outFile << CHISQUARE_THRES << endl;
		int x = cvWaitKey(30);
		if (x == 32)
			pause();
	}//end of while
	int a = cvWaitKey(100000);
	return 0;
}//end of main