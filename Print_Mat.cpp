#include "helper_cv.h"
int frameLearnCounter_L = 0;
int frameLearnCounter_G = 0;
int CHISQUARE_THRES = 0;

int main(int argc, char** argv)
{
	Point pt1, pt2;
	Mat src = imread("images/Bus1.jpg",CV_BGR2GRAY);
	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	threshold(src_gray, src_gray, 25, 255, CV_THRESH_BINARY);
	imshow("Gray",src_gray);
	cvWaitKey(30);
	Vec3b value;
	for (int i = 0; i < src.cols; i++){
		for (int j = 0; j < src.rows; j++){
			//cout << "In" << endl;
			value = src_gray.at<Vec3b>(Point(i, j));
			cout << "Values are " << (int)value.val[0] << "," << (int)value.val[1] << "," << (int)value.val[2] << endl;
			if ((int)value.val[0] == 255){
				cout << "White found at " << "(" << i << "," << j << ")" << endl;
				pause();
			}
		}
	}
	int a = cvWaitKey(0);
	return 0;
}//end of main