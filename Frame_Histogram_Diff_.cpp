#include "helper_cv.h"

/**
* @function main
*/
int main(int argc, char** argv)
{
	Mat src, src2;

	/// Load image
	src = imread("images/histogram/shot0005.png", CV_LOAD_IMAGE_COLOR);
	src2 = imread("images/histogram/shot0012.png", CV_LOAD_IMAGE_COLOR);
	createNMoveWindow("1", max_cols, max_rows,src,ORIGIN+SHIFT,0);
	createNMoveWindow("2", max_cols, max_rows, src2, ORIGIN + SHIFT + WINDISTANCE+max_cols, 0);
	cvWaitKey(30);
	if (!src.data || !src2.data)
	{
		cout << "Unable to load image" << endl;
		pause();
		return -1;
	}

	/// Separate the image in 3 places ( B, G and R )
	vector<Mat> bgr_planes;
	split(src, bgr_planes);

	/// Establish the number of bins
	int histSize = 256;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;
	Mat b_hist2, g_hist2, r_hist2;
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	vector<Mat>hist = getNormalizedHist(src, histSize, range, uniform, accumulate);
	hist[0].copyTo(r_hist);
	hist[1].copyTo(g_hist);
	hist[2].copyTo(b_hist);

	vector<Mat>hist2 = getNormalizedHist(src2, histSize, range, uniform, accumulate);
	hist2[0].copyTo(r_hist2);
	hist2[1].copyTo(g_hist2);
	hist2[2].copyTo(b_hist2);

	//Model parameters
	double b_changeCHI = compareHistograms(b_hist, b_hist2, CHISQUARE);
	if (b_changeCHI < 0)
		b_changeCHI = 0;
	double g_changeCHI = compareHistograms(g_hist, g_hist2, CHISQUARE);

	if (g_changeCHI < 0)
		g_changeCHI = 0;
	double r_changeCHI = compareHistograms(r_hist, r_hist2, CHISQUARE);

	if (r_changeCHI < 0)
		r_changeCHI = 0;

	double changeCHI = B_WEIGHT*b_changeCHI + R_WEIGHT*r_changeCHI + G_WEIGHT*g_changeCHI;

	cout << "Change in CHI RED is: " << r_changeCHI << endl;
	cout << "Change in CHI GREEN is: " << g_changeCHI << endl;
	cout << "Change in CHI BLUE is: " << b_changeCHI << endl;
	cout << "Change in CHI is: " << changeCHI << endl;

	pause();

	return 0;
}