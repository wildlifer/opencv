#include "helper_cv.h"
int frameLearnCounter_L = 0;
int frameLearnCounter_G = 0;
int FRAME_DIFF_THRES = 0;

int main(int argc, char** argv)
{
	//No of Vehicles
	int no_of_vehicles_in_frame = 0;
	int total_vechicles = 0;
	//assuming there will not be more than 5 vehicles in a frame 
	double vehicles_detected[MAX_VEHICLES_IN_FRAME][NO_OF_CUES] = { 0 };
	//File
	ofstream outFile;
	//Video
	VideoCapture cap("videos/5.2.mp4");
	vector<Mat> training(TRAINING_FRAMES);
	vector<Mat> no_contour_frame_vec(NO_CONTOUR_FRAME_COUNT);
	Mat background, gray_background, backup_background;;
	Mat nextframe, gray_nextframe;
	Mat diff_frame;
	outFile.open("data/VehicleCassifier/changeInMetric.txt", 'w');

	for (int i = 0; i < TRAINING_FRAMES; i++){
		cap >> background;
		training[i] = background.clone();
	}

	int THRES = getBackgroundThreshold(training);
	//Canny parameters
	Mat canny_output;
	int thresh = 50;
	int max_thresh = 255;

	//Start here
	cap >> background;
	int frame_width = background.cols;
	int frame_height = background.rows;

	int cropFrameY = frame_height*cropY;
	int cropFrameHeight = frame_height*cropHeight;
	Rect roi(0, cropFrameY, frame_width, cropFrameHeight);
	//get ROI if requested
	if (ROI_ENABLED)
		background = background(roi);
	frame_width = background.cols;
	frame_height = background.rows;

	createNMoveWindow("Original Video", max_cols, max_rows, background, ORIGIN + SHIFT, 0);
	createNMoveWindow("Current background", max_cols, max_rows, background, ORIGIN + SHIFT + WINDISTANCE + max_cols, 0);
	createNMoveWindow("Diff Frame", max_cols, max_rows, background, ORIGIN + SHIFT + WINDISTANCE + max_cols, 350);
	//createNMoveWindow("Detected objects", max_cols, max_rows, background, ORIGIN + SHIFT, 350);
	//Create temporary images
	Mat empty = emptyMat(background);

	CvSize sz = cvGetSize(new IplImage(background));
	cvtColor(background, gray_background, CV_BGR2GRAY);
	Mat bgray(background);
	Mat diff(background);
	int frame_no = 1;
	int no_contour_frame = 0;
	int contour_stuck = 0;
	double area[CONTOUR_STUCK_FRAME_COUNT] = { 0 };
	int contour_point[CONTOUR_STUCK_FRAME_COUNT][2] = { 0 };
	int stuck_frame_count = 0;
	int vehicles = 0;

	while (1) {
		frame_no++;

		//new frame
		cap >> nextframe;
		if (nextframe.empty()) break;

		//get region of interest
		if (ROI_ENABLED){
			nextframe = nextframe(roi).clone();
		}

		// stream used for the conversion
		ostringstream convert;
		convert << frame_no;
		string frameNumberString = convert.str();
		putText(nextframe, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//Show Original Video  - IplImage model
		cvtColor(nextframe, gray_nextframe, CV_BGR2GRAY);
		Mat unblur_gray_nextframe = gray_nextframe.clone();
		medianBlur(gray_nextframe, gray_nextframe, APERTURE);
		imshow("Original Video", nextframe);

		//Show current background
		cvtColor(background, gray_background, CV_BGR2GRAY);
		medianBlur(gray_background, gray_background, APERTURE);
		imshow("Current background", background);

		//Get the difference between background and current frame
		absdiff(gray_nextframe, gray_background, diff);
		medianBlur(diff, diff, APERTURE);
		IplImage *I = new IplImage(diff);
		cvThreshold(I, I, 25, 255, CV_THRESH_BINARY);
		Mat m(I);
		diff = m.clone();
		Mat src = diff.clone();
		vector<vector<Point>> contours;
		Rect boundRec;
		vector<Vec4i> hierarchy;
		findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		Mat noise_less_frame = drawContoursNDeNoise2(diff, contours, MIN_DETECTABLE_AREA);
		//createNMoveWindow("Noise Less", max_cols, max_rows, noise_less_frame, ORIGIN + SHIFT, 350);
		//cvWaitKey(30);
		imshow("Diff Frame", diff);
		
		/***********Background update starts here************/
		Mat new_background = background.clone();
		Scalar value;
		int pixels_updated = 0;
			//cout << "Updating pixels with THRES as " << THRES << endl;
			for (int i = 0; i < background.cols; i++){
				for (int j = 0; j < background.rows; j++){
					//int intensity_diff = abs((int)gray_background.at<uchar>(Point(i, j)) - (int)gray_nextframe.at<uchar>(Point(i, j)));
					double b_diff = abs((int)background.at<cv::Vec3b>(Point(i, j))[0] - (int)nextframe.at<cv::Vec3b>(Point(i, j))[0]);
					double g_diff = abs((int)background.at<cv::Vec3b>(Point(i, j))[1] - (int)nextframe.at<cv::Vec3b>(Point(i, j))[1]);
					double r_diff = abs((int)background.at<cv::Vec3b>(Point(i, j))[2] - (int)nextframe.at<cv::Vec3b>(Point(i, j))[2]);
					//cout << "The difference for point "<<i <<","<<j<< " is: RGB -> " << r_diff << ", " << g_diff << ", " << b_diff << endl;
					int total_diff = (int)(B_WEIGHT*b_diff + G_WEIGHT*g_diff + R_WEIGHT*r_diff);
					//cout << "The total difference for point " << i << "," << j << " is -> " << total_diff << endl;
					//pause();
					if (total_diff < THRES){//originally if (total_diff < MIN_RGB_DIFFERENCE)
						//cout << "R pixel before updation " << (int)background.at<cv::Vec3b>(Point(i, j))[0] << endl;
						new_background.at<cv::Vec3b>(Point(i, j))[0] = nextframe.at<cv::Vec3b>(Point(i, j))[0];
						new_background.at<cv::Vec3b>(Point(i, j))[1] = nextframe.at<cv::Vec3b>(Point(i, j))[1];
						new_background.at<cv::Vec3b>(Point(i, j))[2] = nextframe.at<cv::Vec3b>(Point(i, j))[2];
						//cout << "R pixel after updation " << (int)new_background.at<cv::Vec3b>(Point(i, j))[0] << endl;
						pixels_updated++;
						//pause();
					}//end of if
				}//end of inner for
			}//end of outer for
		

		/*Background cleaning strategy 1: If no contours are reported for a
		continuous NO_CONTOUR_FRAME_COUNT then set the current frame as the background frame*/
		if (no_contour_frame == NO_CONTOUR_FRAME_COUNT){
			THRES = getBackgroundThreshold(no_contour_frame_vec);
			if (THRES < MIN_RGB_DIFFERENCE){
				THRES = MIN_RGB_DIFFERENCE;
				//cout << "Updating Threshold to MIN_RGB_DIFFERENCE as " << THRES << endl;
			}
			background = nextframe.clone();
			backup_background = nextframe.clone();
			//clear the memory
			//cout << "****************Clearing the detected vehicle memory********************" << endl;
			for (int i = 0; i < MAX_VEHICLES_IN_FRAME; i++){
				for (int j = 0; j < NO_OF_CUES; j++){
					vehicles_detected[i][j] = 0;
				}
			}
			vehicles = 0;
			no_contour_frame = 0;
		}
		else{
			//Update the older background with the new one
			background = new_background.clone();
		}

		/***************Check for user input******************/
		int x = cvWaitKey(30);
		if (x == 32)
			pause();
		/***************Check for user input******************/
	}//end of while
	cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
	cout << "Total vehicles detected in the video      " << total_vechicles << endl;
	cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
	pause();
	return 0;
}//end of main