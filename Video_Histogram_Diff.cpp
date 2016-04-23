#include "helper_cv.h"

/**
* @function main
*/
int main(int argc, char** argv)
{
	Mat src, src2;

	/// Load image
	VideoCapture cap("videos/4.5.mp4");
	cap >> src;
	cap >> src;
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Establish the number of bins
	int histSize = 256;

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	vector<Mat>hist = getNormalizedHist(src, histSize, range, uniform, accumulate);
	r_hist=hist[0].clone();
	g_hist=hist[1].clone();
	b_hist=hist[2].clone();
	createNMoveWindow("1", max_cols, max_rows, src, ORIGIN + SHIFT, 0);
	while (1) {
		Mat b_hist2, g_hist2, r_hist2;
		cap >> src2;
		createNMoveWindow("2", max_cols, max_rows, src2, ORIGIN + SHIFT + WINDISTANCE + max_cols, 0);
		cvWaitKey(30);
		if (!src.data || !src2.data)
		{
			cout << "Unable to load data" << endl;
			pause();
			return -1;
		}
		
		vector<Mat>hist2 = getNormalizedHist(src2, histSize, range, uniform, accumulate);
		r_hist2 = hist2[0].clone();
		g_hist2 = hist2[1].clone();
		b_hist2 = hist2[2].clone();

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
	}
	

	return 0;
}