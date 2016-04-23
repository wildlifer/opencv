# include "de.h"
double inibound_h1[] = { 4, 20, 200, 100, 10, 20 };      /* upper parameter bound */
double inibound_l1[] = { 1, 10, 40,  20,  20, 40 };

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
