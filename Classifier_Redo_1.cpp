#include "helper_cv.h"
int frameLearnCounter_L = 0;
int frameLearnCounter_G = 0;
int CHISQUARE_THRES = 0;

int main(int argc, char** argv)
{
	//No of Vehicles
	int no_of_vehicles = 0;
	vector<vector<double>> vehicles;
	//File
	ofstream outFile;
	//Video
	VideoCapture cap("videos/5.2.mp4");
	Mat background, gray_background;
	Mat nextframe, gray_nextframe;
	Mat diff_frame;
	outFile.open("data/VehicleCassifier/changeInMetric.txt", 'w');
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
	width = background.cols;
	height = background.rows;

	//Create temporary images
	CvSize sz = cvGetSize(new IplImage(background));
	cvtColor(background, gray_background, CV_BGR2GRAY);
	Mat bgray(background);
	Mat diff(background);
	int frame_no = 1;
	while (1) {
		frame_no++;
		//new frame
		cap >> nextframe;
		if (nextframe.empty()) break;
		//get region of interest
		if (ROI_ENABLED){
			nextframe = nextframe(roi);
		}

		//Show Original Video  - IplImage model
		cvtColor(nextframe, gray_nextframe, CV_BGR2GRAY);
		medianBlur(gray_nextframe, gray_nextframe, APERTURE);
		createNMoveWindow("Original Video", max_cols, max_rows, gray_nextframe, ORIGIN + SHIFT, 0);

		//Show current background
		medianBlur(gray_background, gray_background, APERTURE);
		createNMoveWindow("Current background", max_cols, max_rows, gray_background, ORIGIN + SHIFT + WINDISTANCE + max_cols, 0);

		//Get the difference between background and current frame
		absdiff(gray_nextframe, gray_background, diff);
		medianBlur(diff, diff, APERTURE);
		IplImage *I = new IplImage(diff);
		cvThreshold(I, I, 25, 255, CV_THRESH_BINARY);
		Mat m(I);
		diff = m.clone();
		createNMoveWindow("Diff Frame", max_cols, max_rows, diff, ORIGIN + SHIFT + WINDISTANCE + max_cols, 350);

		Mat src = diff.clone();
		Mat original = gray_background.clone();
		vector<vector<Point>> contours;
		Rect boundRec;
		vector<Vec4i> hierarchy;
		//Canny(src, canny_output, thresh, thresh * 2, 3);
		//CV_RETR_EXTERNAL is important here. It helps to find only outer contours which is what we need
		findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		Mat noise_less_frame = drawContoursNDeNoise2(diff, contours, MIN_DETECTABLE_AREA);
		createNMoveWindow("Noise Less", max_cols, max_rows, noise_less_frame, ORIGIN + SHIFT, 350);
		cvWaitKey(30);
		findContours(noise_less_frame, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		src = noise_less_frame.clone();
		//findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		vector<Rect> boundRect(contours.size());
		Point pt1, pt2;
		//cout << "No of contours found: " << contours.size() << endl;
		double max_area = 0;
		for (int i = 0; i < contours.size(); i++){
			if (contourArea(contours[i], false)>max_area){
				max_area = contourArea(contours[i]);
				boundRec = boundingRect((contours[i]));
			}
		}

		//background update starts here
		Mat new_background(gray_background.size(), CV_8U);
		Scalar value;
		int pixels_updated = 0;
		for (int i = 0; i < background.cols; i++){
			for (int j = 0; j < background.rows; j++){
				int intensity_diff = abs((int)gray_background.at<uchar>(Point(i, j)) - (int)gray_nextframe.at<uchar>(Point(i, j)));
				if (intensity_diff < MIN_INTENSITY_DIFFERENCE){
					//cout << "Intensity difference found at " << i << "," << j << endl;
					//cout << "Intensity in background is " << (int)gray_background.at<uchar>(Point(i, j)) << endl;
					//cout << "Intensity in newframe is " << (int)gray_nextframe.at<uchar>(Point(i, j)) << endl;
					new_background.at<uchar>(Point(i, j)) = (int)gray_nextframe.at<uchar>(Point(i, j));
					pixels_updated++;

				}
				else if (intensity_diff > MIN_INTENSITY_DIFFERENCE && intensity_diff < MIN_VEHICLE_IDENTIFIABLE_DIFFERENCE){
					//cout << "Foreground pixel detected at " <<i<<","<<j<<" with intensity difference: " << intensity_diff << endl;
					new_background.at<uchar>(Point(i, j)) = (int)gray_nextframe.at<uchar>(Point(i, j));
					pixels_updated++;
				}
				else if (intensity_diff > MIN_VEHICLE_IDENTIFIABLE_DIFFERENCE){
					//cout << "CAR pixel detected at " << i << "," << j << " with intensity difference: " << intensity_diff << endl;
					new_background.at<uchar>(Point(i, j)) = (int)gray_background.at<uchar>(Point(i, j));
					//pause();
				}
			}
		}
		//Update the older background with the new one
		gray_background = new_background.clone();
		if (frame_no % PAUSE_AFTER_FRAMES == 0){
			cout << "Processed frame no: " << frame_no << endl;
			cout << "No of pixels updated: " << pixels_updated << endl;
			pause();
		}

		int x = cvWaitKey(30);
		if (x == 32)
			pause();
	}//end of while
	int a = cvWaitKey(100000);
	return 0;
}//end of main