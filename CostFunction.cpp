# include "de.h"

double inibound_l1[] = { 1, 5,  10,   8, 2, 4 };			/* lower parameter bound */
double inibound_h1[] = { 2, 20, 100, 20, 4, 6 };      /* upper parameter bound */

/*[inverseResolution,src_gray.rows / 3, cannyThreshold, accumulatorThreshold, minRadius, maxRadius]*/
/*
dp – Inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
minDist – Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
param1 – First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
param2 – Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
minRadius – Minimum circle radius.
maxRadius – Maximum circle radius.
*/

using namespace std;
using namespace cv;
vector<Vec3f> evaluate(Mat src, Mat src_gray,int D, double tmp[], long *nfeval)
{
	/* polynomial fitting problem */
	double result = 0;
	// will hold the results of the detection
	std::vector<Vec3f> circles;
	for (int j = 0; j <6; j++){
		if (tmp[j] < inibound_l1[j])
			tmp[j] = inibound_l1[j];
		if (tmp[j] > inibound_h1[j])
			tmp[j] = inibound_h1[j];
	}
	// runs the actual detection
	//HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 8, cannyThreshold, accumulatorThreshold, 14, 40);
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 
		tmp[0],
		tmp[1],
		tmp[2],
		tmp[3],
		tmp[4],
		(int)tmp[5]);
	//cout << "No of Detected circles are " << circles.size() <<endl;
	return circles;
}
