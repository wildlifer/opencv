#include "helper_cv.h"
int frameLearnCounter_L = 0;
int frameLearnCounter_G = 0;
int CHISQUARE_THRES = 0;

int main(int argc, char** argv)
{
	Point pt1, pt2;
	Mat img = imread("images/Bus1.jpg");
	//imshow("Original image",img);
	createNMoveWindow("Original Image", max_cols, max_rows, img, ORIGIN + SHIFT + WINDISTANCE , 0);
	//Mat clone = img.clone();
	CvSize sz = cvGetSize(new IplImage(img));
	Mat clone = img.clone();
	Mat clone_gray(clone);
	cvtColor(clone, clone_gray, CV_BGR2GRAY);
	medianBlur(clone_gray, clone_gray, CV_MEDIAN);
	threshold(clone_gray, clone_gray, 100, 255, CV_THRESH_BINARY);
	createNMoveWindow("Thresholded Image", max_cols, max_rows, clone_gray, ORIGIN + SHIFT + WINDISTANCE + max_cols, 0);
	cvWaitKey(30);
	//find contours
	vector<vector<Point>> contours;
	Rect boundRec;
	vector<Vec4i> hierarchy;
	
	findContours(clone_gray, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	if (contours.size() == 0){
		cout << "No contours found" << endl;
		pause();
	}
	drawContoursNDeNoise1(clone_gray, contours,300);
	//pt1.x = 50;
	//pt1.y = 100;
	//for (int k = 0; k < pt1.y; k++){
	//	for (int l = 0; l < pt1.x; l++){
	//		//src_clone.data[src_clone.step[0] * i + src_clone.step[1] * j + 0] = 0;
	//		//cout << "Updating pixel with 0" << endl;
	//		clone.at<Vec3b>(Point(l, k)) = 0;
	//	}
	//}
	//imshow("Un-noised image", clone);
	int a = cvWaitKey(0);
	return 0;
}//end of main