#include "helper_cv.h"
void pause(){
	int xx;
	cin >> xx;
	cin.clear();
	cin.ignore(INT_MAX, '\n');
}
double getBackgroundThreshold(vector<Mat> seq){
	double thres = 0;
	vector<double> diffs(seq.size() - 1);
	int total_diff = 0;
	Scalar value;
	Mat first_frame = seq[0].clone();

	for (int it = 1; it < seq.size();it++){	
		Mat nextframe = seq[it].clone();
		for (int i = 0; i < first_frame.cols; i++){
			for (int j = 0; j < first_frame.rows; j++){
				//cout << "B: " << (int)first_frame.at<cv::Vec3b>(Point(i, j))[0] << endl;
				//cout << "B: " << (int)nextframe.at<cv::Vec3b>(Point(i, j))[0] << endl;;
				double b_diff = abs((int)first_frame.at<cv::Vec3b>(Point(i, j))[0] - (int)nextframe.at<cv::Vec3b>(Point(i, j))[0]);
				double g_diff = abs((int)first_frame.at<cv::Vec3b>(Point(i, j))[1] - (int)nextframe.at<cv::Vec3b>(Point(i, j))[1]);
				double r_diff = abs((int)first_frame.at<cv::Vec3b>(Point(i, j))[2] - (int)nextframe.at<cv::Vec3b>(Point(i, j))[2]);
				//cout << "The difference for point "<<i <<","<<j<< " is: RGB -> " << r_diff << ", " << g_diff << ", " << b_diff << endl;
				total_diff += (int)(B_WEIGHT*b_diff + G_WEIGHT*g_diff + R_WEIGHT*r_diff);
			}
		}
		//cout << "Total diff is " << total_diff << endl;
		double a = nextframe.cols*nextframe.rows;
		diffs[it - 1] = total_diff/a;
		//cout << "Actual diff is " << diffs[it - 1] << endl;
		first_frame = nextframe.clone();
	}
	
	thres = mean(diffs);
	double dev = stdev(thres,diffs);
	cout << "Threshold is ---- " << (thres + dev) << endl;
	//pause();
	return (thres+dev);
}
double stdev(double mean, vector<double> arr){
	double std = 0;
	for (int i = 0; i<arr.size(); ++i)
		std += (arr[i] - mean)*(arr[i] - mean);

	return sqrt(std/arr.size());
}
double mean(vector<double> arr){
	double mean = 0;
	for (int i = 0; i < (arr).size(); i++){
		mean += arr[i];
	}
	mean = mean / (arr).size();
	return mean;
}
double mean(double arr[]){
	double mean = 0;
	for (int i = 0; i < sizeof(arr); i++){
		mean += arr[i];
	}
	mean = mean / sizeof(arr);
	return mean;
}
Mat emptyMat(Mat i){
	Mat image(i.rows, i.cols, CV_8UC3, Scalar(0, 0, 0));
	return image;
}
int getBackgroundThreshold(VideoCapture cap, int limit){
	//threshold to be returned
	int thr;
	//window dispay parameters
	int max_rows = 300;
	int max_cols = 400;
	//storage for histogram differences
	vector<int> v(limit, 0);
	Mat background, nextframe;
	vector<Mat> bgr_planes_curr_frame;
	vector<Mat> bgr_planes_background;
	/// Establish the number of bins
	int histSize = 256;
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;

	//Start here
	cap >> background;
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
	while (i < limit) {
		//new frame
		cap >> nextframe;
		if (nextframe.empty()) break;
		if (first){
			nextframe.copyTo(background);
			first = false;
			if (ROI_ENABLED)
				background = background(roi);
			
			createNMoveWindow("Background Video", max_cols, max_rows, background, 0, 0);
			
			vector<Mat>hist = getNormalizedHist(background, histSize, range, uniform, accumulate);
			hist[0].copyTo(r_hist_back);
			hist[1].copyTo(g_hist_back);
			hist[2].copyTo(b_hist_back);
			////cout << "Continuing" << endl;
			continue;
		}
		//get region of interest if requested
		if (ROI_ENABLED)
			nextframe = nextframe(roi);
		createNMoveWindow("Training Video", max_cols, max_rows, nextframe, ORIGIN + SHIFT, 0);
		vector<Mat>hist = getNormalizedHist(nextframe, histSize, range, uniform, accumulate);
		hist[0].copyTo(r_hist_curr);
		hist[1].copyTo(g_hist_curr);
		hist[2].copyTo(b_hist_curr);

		//Model parameters
		double b_changeCHI = compareHistograms(b_hist_back, b_hist_curr, CHISQUARE);
		double g_changeCHI = compareHistograms(g_hist_back, g_hist_curr, CHISQUARE);
		double r_changeCHI = compareHistograms(r_hist_back, r_hist_curr, CHISQUARE);
		if (b_changeCHI < 0)
			b_changeCHI = 0;

		if (g_changeCHI < 0)
			g_changeCHI = 0;

		if (r_changeCHI < 0)
			r_changeCHI = 0;

		double changeCHI   = B_WEIGHT*b_changeCHI + R_WEIGHT*r_changeCHI + G_WEIGHT*g_changeCHI;
		cout << "Change in  Chi is " << changeCHI << endl;
		
		if (changeCHI >0){
			//cout << "Updating" << endl;
			Mat b_hist_back, g_hist_back, r_hist_back;/*This statement  saved me*/
			//this can be optimized
			Mat r(r_hist_curr);
			Mat g(g_hist_curr);
			Mat b(b_hist_curr);
			r.copyTo(r_hist_back);
			g.copyTo(g_hist_back);
			b.copyTo(b_hist_back);
		}
		v[i] = (int)changeCHI;
		i++;
		bgr_planes_curr_frame.clear();
		int xx = cvWaitKey(10);
	}
	thr = getThrs(v);
	//cout << "***************************************" << endl;
	//cout << " Threshold is: " << thr << endl;
	//cout << "***************************************" << endl;
	//pause();
	cvDestroyWindow("Training Video");
	cvDestroyWindow("Background Video");
	return thr;
}

int getThrs(vector<int> v){
	if (v.size() == 0){
		return 0;
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
	return (int)mean + (int)stdev;
}
void createNMoveWindow(char name[], int width, int height, Mat data, int x, int y){
	cvNamedWindow(name, CV_WINDOW_NORMAL);
	cvResizeWindow(name, max_cols, max_rows);
	cvShowImage(name, new IplImage(data));
	cvMoveWindow(name, x, y);
	//cv::waitKey(30);
}
vector<Mat> getNormalizedHist(Mat frame, int histSize, float range[], bool uniform, bool accumulate){
	Mat b_hist, g_hist, r_hist;
	vector<Mat> hist;
	vector<Mat> bgr_planes;
	split(frame, bgr_planes);
	const float* histRange = { range };

	//background frame histogram
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	hist.push_back(r_hist);
	hist.push_back(g_hist);
	hist.push_back(b_hist);

	return hist;
}
double compareHistograms(Mat a, Mat b, int measure){
	double diff;
	diff = compareHist(a, b, measure);

	return diff;
}
void drawContours(Mat src, vector<vector<Point>> contours, double max_area,  char* winName){
	vector<Rect> boundRect(contours.size());
	Point pt1, pt2;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i], false);
		if (area>max_area){
			//cout << "rect no " << boundRect[i] << endl;
			boundRect[i] = boundingRect(Mat(contours[i]));
			////cout << "rect no " << boundRect[i] << endl;
			pt1.x = boundRect[i].x;
			pt1.y = boundRect[i].y;
			pt2.x = boundRect[i].x + boundRect[i].width;
			pt2.y = boundRect[i].y + boundRect[i].height;
			rectangle(src, pt1, pt2, Scalar(255), 1);
			//createNMoveWindow(winName, max_cols, max_rows, src, ORIGIN + SHIFT, 350);
			//imshow(winName, src);
			//cvWaitKey(30);
		}//end of if
	}//end of for
}

void drawContours(Mat src, vector<vector<Point>> contours){
	vector<Rect> boundRect(contours.size());
	Point pt1, pt2;
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i], false);	
		//cout << "Contour area " << area << endl;
		boundRect[i] = boundingRect(Mat(contours[i]));
		pt1.x = boundRect[i].x;
		pt1.y = boundRect[i].y;
		pt2.x = boundRect[i].x + boundRect[i].width;
		pt2.y = boundRect[i].y + boundRect[i].height;
		rectangle(src, pt1, pt2, Scalar(255), 1);
		createNMoveWindow("Contours", max_cols, max_rows, src, ORIGIN + SHIFT, 350);
	}//end of for
}
void drawContoursNDeNoise(Mat src, vector<vector<Point>> contours, double min_area){
	vector<Rect> boundRect(contours.size());
	Point pt1, pt2;
	Mat src_clone = src.clone();
	Scalar value;
	bool white = false;
	createNMoveWindow("Before Removed Noise", max_cols, max_rows, src, ORIGIN + SHIFT, 350);

	for (int i = 0; i < src_clone.cols; i++){
		for (int j = 0; j < src_clone.rows; j++){
			////cout << "In" << endl;
			value = src_clone.at<uchar>(Point(i, j));
			////cout << "Values are " << (int)value.val[0] << " at " <<i << ","<<j<< endl;
			if ((int)value.val[0] == 1){
				////cout << "White found at " << "(" << i << "," << j << ")" << endl;
				src.at<uchar>(Point(i, j)) = 0;
				white = true;
			}
		}
	}

	//for (int i = 0; i < contours.size(); i++)
	//{
	//	double area = contourArea(contours[i], false);
	//	boundRect[i] = boundingRect(Mat(contours[i]));
	//	////cout << "rect no " << boundRect[i] << endl;
	//	pt1.x = boundRect[i].x;
	//	pt1.y = boundRect[i].y;
	//	pt2.x = boundRect[i].x + boundRect[i].width;
	//	pt2.y = boundRect[i].y + boundRect[i].height;
	//	//cout << "Contour Found at " << "(" << pt1.x << ", " << pt1.y << ")" << "with area "<< area<<endl;
	//	//findOnes(src_clone);
	//	if ((area > min_area) ){
	//		rectangle(src, pt1, pt2, Scalar(255), 1);
	//		//createNMoveWindow("Detected Vehicles", max_cols, max_rows, src, ORIGIN + SHIFT, 350);
	//	}
	//	else if (area <= min_area){
	//		Vec3b v;
	//		//cout << "Noise Found at "<< "("<<pt1.x <<", "<<pt1.y << ")" << " with width "<< pt2.x<< " and height "<<pt2.y<<endl;
	//		//createNMoveWindow("Detected Noise", max_cols, max_rows, src, ORIGIN + SHIFT+40, 350);
	//		for (int k = pt1.y; k < pt1.y+pt2.y; k++){
	//			for (int l = pt1.x; l < pt1.x+pt2.x; l++){
	//				v = src_clone.at<Vec3b>(Point(k + 1, l + 1));
	//				//cout << "Value at " << k+1 << "," << l+1 << " is " << src_clone.at<Vec3b>(Point(k+1, l+1))<<endl;
	//				//src_clone.data[src_clone.step[0] * i + src_clone.step[1] * j + 0] = 0;
	//				//cout << "Updating pixel with 0 at "<< k <<","<<l << endl;
	//				src_clone.at<Vec3b>(Point(k, l)) = 0;
	//				//cout << "Value after updation  " <<src_clone.at<Vec3b>(Point(k, l)) << endl;
	//				pause();
	//			}
	//		}
	//		createNMoveWindow("Removing Noise", max_cols, max_rows, src_clone, ORIGIN, 350);
	//		cvWaitKey(30);
	//		pause();
	//		break;
	//	}//end of if
	//}//end of for
	createNMoveWindow("Removed Noise", max_cols, max_rows, src, ORIGIN, 350);
	cvWaitKey(30);
	if (white == true)
		pause();
	//pause();
}

void drawContoursNDeNoise1(Mat src, vector<vector<Point>> contours, double min_area){
	vector<Rect> boundRect(contours.size());
	Point pt1, pt2;
	Mat src_clone = src.clone();
	Scalar value;
	bool white = false;
	createNMoveWindow("Before Removed Noise", max_cols, max_rows, src, ORIGIN+SHIFT, 350);
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i], false);
		boundRect[i] = boundingRect(Mat(contours[i]));
		pt1.x = boundRect[i].x;
		pt1.y = boundRect[i].y;
		pt2.x = boundRect[i].x + boundRect[i].width;
		pt2.y = boundRect[i].y + boundRect[i].height;
		if ((area > min_area) ){
			//rectangle(src, pt1, pt2, Scalar(255), 1);
			//createNMoveWindow("Detected Vehicles", max_cols, max_rows, src, ORIGIN + SHIFT, 350);
		}
		else if (area <= min_area){
			//cout << "Noise Found at " << "(" << pt1.x << ", " << pt1.y << ")" << " with width " << boundRect[i].width << " and height " << boundRect[i].height << endl;
			//createNMoveWindow("Detected Noise", max_cols, max_rows, src, ORIGIN + SHIFT+40, 350);
			for (int k = pt1.x; k < pt2.x; k++){
				for (int l = pt1.y; l < pt2.y; l++){
					value = (int)src.at<uchar>(Point(k,l));
					////cout << "Value at " << k << "," << l << " is " << value << endl;
					if (value.val[0] > 0){					
						src.at<uchar>(Point(k, l)) = 0;
						////cout << "Updating pixel with 0 at " << k << "," << l << endl;
						////cout << "Value after updation  " <<src_clone.at<uchar>(Point(k, l)) << endl;
						//pause();
					}
				}
			}
			createNMoveWindow("Removing Noise", max_cols, max_rows, src, ORIGIN, 350);
			cvWaitKey(30);
		}//end of if

	}
	createNMoveWindow("Removed Noise", max_cols, max_rows, src, ORIGIN, 350);
	cvWaitKey(30);
		pause();
	//pause();
}

Mat drawContoursNDeNoise2(Mat src, vector<vector<Point>> contours, double min_area){
	vector<Rect> boundRect(contours.size());
	Point pt1, pt2;
	Mat src_clone = src.clone();
	Scalar value;
	bool white = false;
	//createNMoveWindow("Before Removed Noise", max_cols, max_rows, src, ORIGIN + SHIFT, 350);
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i], false);
		boundRect[i] = boundingRect(Mat(contours[i]));
		pt1.x = boundRect[i].x;
		pt1.y = boundRect[i].y;
		pt2.x = boundRect[i].x + boundRect[i].width;
		pt2.y = boundRect[i].y + boundRect[i].height;
		//cout << "Area: " << area << endl;
		if ((area > min_area)){
			//rectangle(src, pt1, pt2, Scalar(255), 1);
			//createNMoveWindow("Detected Vehicles", max_cols, max_rows, src, ORIGIN + SHIFT, 350);
		}
		else if (area <= min_area){
			//cout << "Noise Found at " << "(" << pt1.x << ", " << pt1.y << ")" << " with width " << boundRect[i].width << " and height " << boundRect[i].height << endl;
			white = true;
			//createNMoveWindow("Detected Noise", max_cols, max_rows, src, ORIGIN + SHIFT+40, 350);
			for (int k = pt1.x; k < pt2.x; k++){
				for (int l = pt1.y; l < pt2.y; l++){
					value = (int)src.at<uchar>(Point(k, l));
					////cout << "Value at " << k << "," << l << " is " << value << endl;
					if (value.val[0] > 0){
						src.at<uchar>(Point(k, l)) = 0;
					}
				}
			}
		//	createNMoveWindow("Removing Noise", max_cols, max_rows, src, ORIGIN, 350);
			//cvWaitKey(30);
		}//end of if

	}

	//createNMoveWindow("Removed Noise", max_cols, max_rows, src, ORIGIN, 350);
	//cvWaitKey(30);
	//if (white)
		//pause();
	return src;
}
Mat drawContoursNDeNoise2Coloured(Mat src, Mat pixel_src, vector<vector<Point>> contours, double min_area){
	vector<Rect> boundRect(contours.size());
	Point pt1, pt2;
	Mat src_clone = src.clone();
	Scalar value;
	bool white = false;
	if (contours.size() == 0){
		cout << "No contours found in color frame" << endl;
		return src;
	}
	//createNMoveWindow("Before Removed Noise", max_cols, max_rows, src, ORIGIN + SHIFT, 350);
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i], false);
		boundRect[i] = boundingRect(Mat(contours[i]));
		pt1.x = boundRect[i].x;
		pt1.y = boundRect[i].y;
		pt2.x = boundRect[i].x + boundRect[i].width;
		pt2.y = boundRect[i].y + boundRect[i].height;
		cout << "Area: " << area << endl;
		if ((area > min_area)){
			//rectangle(src, pt1, pt2, Scalar(255), 1);
			//createNMoveWindow("Detected Vehicles in color frame", max_cols, max_rows, src, ORIGIN + SHIFT, 350);
		}
		else if (area <= min_area){
			//cout << "Noise Found at " << "(" << pt1.x << ", " << pt1.y << ")" << " with width " << boundRect[i].width << " and height " << boundRect[i].height << endl;
			for (int k = pt1.x; k < pt2.x; k++){
				for (int l = pt1.y; l < pt2.y; l++){
					src.at<cv::Vec3b>(Point(k, l))[0] = pixel_src.at<cv::Vec3b>(Point(k, l))[0];
					src.at<cv::Vec3b>(Point(k, l))[1] = pixel_src.at<cv::Vec3b>(Point(k, l))[1];
					src.at<cv::Vec3b>(Point(k, l))[2] = pixel_src.at<cv::Vec3b>(Point(k, l))[2];
					cout << "Background cleaned at " << k << "," << l << endl;
				}
			}
			pause();
		}//end of if

	}
	return src;
}
void findOnes(Mat src){
	Scalar value;

	for (int i = 0; i < src.cols; i++){
		for (int j = 0; j < src.rows; j++){
			value = src.at<uchar>(Point(i, j));
			if ((int)value.val[0]==1 ){
				//cout << "White found at " << "(" << i << "," << j << ")" << endl;
				pause();
			}
		}
	}
}

bool checkCircles(Mat src_gray){
	bool found = false;
	found = HoughDetection(src_gray, src_gray, cannyThresholdInitialValue, accumulatorThresholdInitialValue, inverseResolutionInitialValue, minRadius, maxRadius);
	return found;
}
bool HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold, int inverseResolution, int min, int max)
{
	// will hold the results of the detection
	std::vector<Vec3f> circles;
	// runs the actual detection
	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 3, cannyThreshold, accumulatorThreshold, min, max);

//	imshow("Circles1", src_gray);
	//cvWaitKey(30);
	if (circles.size() <= 0){
		cout << "No circles found" << endl;
		return false;
	}
	/// Apply the Hough Transform to find the circles
	//HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1.98, src_gray.rows / 5, 250, 50, 0, 30);

	// clone the colour, input image for displaying purposes
	Mat display = src_display.clone();
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(display, center, 1, Scalar(255, 255, 0), -1, 8, 0);
		// circle outline
		circle(display, center, radius, Scalar(255, 255, 255), 1, 8, 0);
	}
	cout << circles.size() << " circles found" << endl;
	// shows the results
	imshow("Circles", display);
	cvWaitKey(30);
	return true;
}

