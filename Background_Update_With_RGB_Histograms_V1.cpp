#include "helper_cv.h"
int frameLearnCounter_L = 0;
int frameLearnCounter_G = 0;
int CHISQUARE_THRES = 0;

int main(int argc, char** argv)
{
	VideoCapture cap("videos/4.4.mp4");
	vector<Mat> initFrames;
	ofstream outFile;
	int threshold = getBackgroundThreshold(cap, INITIAL_LEARN_FRAME_LIMIT);
	CHISQUARE_THRES = threshold;;
	Mat background, hsv_background, gray_background;
	Mat nextframe, hsv_nextframe, gray_nextframe;
	Mat diff_frame;
	outFile.open("data/VehicleCassifier/changeInMetric.txt", 'w');

	/// Separate the image in 3 places ( B, G and R )
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
	//split(background, bgr_planes_background);
	//background frame histogram
	//calcHist(&bgr_planes_background[0], 1, 0, Mat(), b_hist_back, 1, &histSize, &histRange, uniform, accumulate);
	//calcHist(&bgr_planes_background[1], 1, 0, Mat(), g_hist_back, 1, &histSize, &histRange, uniform, accumulate);
	//calcHist(&bgr_planes_background[2], 1, 0, Mat(), r_hist_back, 1, &histSize, &histRange, uniform, accumulate);
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	/// Normalize the result to [ 0, histImage.rows ]
	//normalize(b_hist_back, b_hist_back, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//normalize(g_hist_back, g_hist_back, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//normalize(r_hist_back, r_hist_back, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	vector<Mat>hist = getNormalizedHist(background, histSize, range, uniform, accumulate);
	hist[0].copyTo(r_hist_back);
	hist[1].copyTo(g_hist_back);
	hist[2].copyTo(b_hist_back);
	//Create temporary images
	CvSize sz = cvGetSize(new IplImage(background));
	IplImage* oldFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);;
	IplImage* frame = cvCreateImage(sz, IPL_DEPTH_8U, 3);;
	IplImage* diffFrameBinary = cvCreateImage(sz, IPL_DEPTH_8U, 3);;

	IplImage* agray = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	IplImage* bgray = cvCreateImage(sz, IPL_DEPTH_8U, 1);;

	IplImage* abin = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	IplImage* bbin = cvCreateImage(sz, IPL_DEPTH_8U, 1);;
	IplImage* diff = cvCreateImage(sz, IPL_DEPTH_8U, 1);;

	cvNamedWindow("Diff Frame", 0);
	int ii = 0;
	//store previous images
	vector<Mat> matStore(LEARN_FRAME_LIMIT_LESS_THRES + 1);
	vector<int> hist_diff_store(LEARN_FRAME_LIMIT_LESS_THRES);
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
		/*split(nextframe, bgr_planes_curr_frame);
		/// Compute the histograms:
		calcHist(&bgr_planes_curr_frame[0], 1, 0, Mat(), b_hist_curr, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes_curr_frame[1], 1, 0, Mat(), g_hist_curr, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes_curr_frame[2], 1, 0, Mat(), r_hist_curr, 1, &histSize, &histRange, uniform, accumulate);
		/// Normalize the result to [ 0, histImage.rows ]
		normalize(b_hist_curr, b_hist_curr, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(g_hist_curr, g_hist_curr, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(r_hist_curr, r_hist_curr, 0, histImage.rows, NORM_MINMAX, -1, Mat());
*/
		//Show Original Video  - IplImage model
		frame = new IplImage(nextframe);
		cvCvtColor(frame, bgray, CV_BGR2GRAY);
		createNMoveWindow("Original Video", max_cols, max_rows, frame, ORIGIN + SHIFT, 0);

		//Show current background
		oldFrame = new IplImage(background);
		cvCvtColor(oldFrame, agray, CV_BGR2GRAY);
		createNMoveWindow("Current background", max_cols, max_rows, oldFrame, ORIGIN + SHIFT + WINDISTANCE + max_cols, 0);

		//Get the difference between background and current frame
		cvAbsDiff(agray, bgray, diff);
		cvSmooth(diff, diff, CV_BLUR);
		cvThreshold(diff, diff, 25, 255, CV_THRESH_BINARY);
		
		/*absdiff(background, nextframe, diff_frame);*/
		createNMoveWindow("Diff Frame", max_cols, max_rows, diff, ORIGIN + SHIFT + WINDISTANCE + max_cols, 350);
		
		//Model parameters
		double b_changeCHI = compareHistograms(b_hist_back, b_hist_curr, CHISQUARE);
		double g_changeCHI = compareHistograms(g_hist_back, g_hist_curr, CHISQUARE);
		double r_changeCHI = compareHistograms(r_hist_back, r_hist_curr, CHISQUARE);		
		double changeCHI = B_WEIGHT*b_changeCHI + R_WEIGHT*r_changeCHI + G_WEIGHT*g_changeCHI;
		
		cout << "Change in Chi is " << (changeCHI) << endl;

		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		int i = 1;

		Mat src(diff);
		Mat original(bgray);
		vector<vector<Point> > contours;
		Rect boundRec;
		vector<Vec4i> hierarchy;
		//Canny(src, canny_output, thresh, thresh * 2, 3);
		//CV_RETR_EXTERNAL is important here. It helps to find only outer contours which is what we need
		findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
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
		//Threshold comparison begins here
		if (changeCHI < CHISQUARE_THRES){
			//Store the difference
			if (frameLearnCounter_L < LEARN_FRAME_LIMIT_LESS_THRES){
				//cout << "Inserting into hist_diff_store in < THRES at index " << frameLearnCounter_L << endl;
				hist_diff_store.push_back((int)changeCHI);
				//cout << "After insertion into hist_diff_store in < THRES at index " << frameLearnCounter_L << endl;
				//cout << "Inserting in hist_diff_store value : " << changeCHI << endl;
				//pause();
			}
			//Re-initialize the matrix. Very important
			Mat b_hist_back, g_hist_back, r_hist_back;
			////New Background
			frameLearnCounter_L++;
			if (frameLearnCounter_L <= LEARN_FRAME_LIMIT_LESS_THRES){
				//cout << "Updating frameLearnCounter_L to:" << frameLearnCounter_L << " in < THRES" << endl;
			}
			//cout << "Below Threshold with frame counter: " << frameLearnCounter << " and no. of contours: " << contours.size() << endl;
			//Keep previous copies of similar images
			//if (frameLearnCounter<LEARN_FRAME_LIMIT)
			//nextframe.copyTo(matStore[frameLearnCounter]);
			if (contours.size() == 0 && frameLearnCounter_L == LEARN_FRAME_LIMIT_LESS_THRES){
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;
				cout << "IN < THRES and condition 1" << endl;
				background = nextframe.clone();
				r_hist_back = r_hist_curr.clone();
				g_hist_back = g_hist_curr.clone();
				b_hist_back = b_hist_curr.clone();
				//cout << "Updating Background and changing Threshold to: " << *std::max_element(begin(hist_diff_store), end(hist_diff_store)) << endl;
				//CHISQUARE_THRES = *std::max_element(begin(hist_diff_store), end(hist_diff_store));
				cout << "Contour area less than MIN_AREA. Updating Background and changing Threshold to: " << changeCHI << endl;
				CHISQUARE_THRES = changeCHI;
				hist_diff_store.clear();
				//pause();
			}
			else if (contours.size()>0 && max_area<MIN_DETECTABLE_AREA && frameLearnCounter_L == LEARN_FRAME_LIMIT_LESS_THRES){
				cout << "IN < THRES and condition 2" << endl;
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;
				//matStore.clear();
				//copy current frame to the background
				background = nextframe.clone();
				r_hist_back = r_hist_curr.clone();
				g_hist_back = g_hist_curr.clone();
				b_hist_back = b_hist_curr.clone();
				//cout << "Contour area less than MIN_AREA. Updating Background and changing Threshold to: " << *std::max_element(begin(hist_diff_store), end(hist_diff_store)) << endl;
				//CHISQUARE_THRES = *std::max_element(begin(hist_diff_store), end(hist_diff_store));
				cout << "Contour area less than MIN_AREA. Updating Background and changing Threshold to: " << changeCHI << endl;
				CHISQUARE_THRES = changeCHI;
				hist_diff_store.clear();
				//pause();
			}
			else if (contours.size()>0 && max_area>MIN_DETECTABLE_AREA && max_area<MIN_VEHICLE_AREA && frameLearnCounter_L == LEARN_FRAME_LIMIT_LESS_THRES){
				cout << "IN < THRES and condition 3" << endl;
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;
				hist_diff_store.clear();
				//pause();
			}
			else if (contours.size()>0 && max_area > MIN_VEHICLE_AREA && frameLearnCounter_L == LEARN_FRAME_LIMIT_LESS_THRES){
				cout << "IN < THRES and condition 4" << endl;
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;
				hist_diff_store.clear();
				//pause();
			}

		}//end of if(changeCHI < CHISQUARE_THRES)
		else if (changeCHI>CHISQUARE_THRES){
			frameLearnCounter_G++;
			if (frameLearnCounter_G <= LEARN_FRAME_LIMIT_GREATER_THRES){
				//cout << "Updating frameLearnCounter_G to:" << frameLearnCounter_G << " in > THRES" << endl;
			}
			if (frameLearnCounter_G > LEARN_FRAME_LIMIT_GREATER_THRES){
				frameLearnCounter_G = 0;
				hist_diff_store.clear();
			}
			//cout << "Above Threshold with frame counter: " << frameLearnCounter_G << endl;
			//matStore.clear();
			if (contours.size() == 0 && frameLearnCounter_G == LEARN_FRAME_LIMIT_GREATER_THRES){
				cout << "IN > THRES and condition 1" << endl;
				//Re-initialize the matrix. Very important
				Mat b_hist_back, g_hist_back, r_hist_back;
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;
				//copy current frame to the background
				background = nextframe.clone();
				r_hist_back = r_hist_curr.clone();
				g_hist_back = g_hist_curr.clone();
				b_hist_back = b_hist_curr.clone();
				cout << "Updating Background as no contours found" << endl;
				cout << "*****************************************" << endl;
				cout << "Updating CHISQUARE_THRES to: " << changeCHI << endl;
				cout << "*****************************************" << endl;
				CHISQUARE_THRES = changeCHI;
				hist_diff_store.clear();
				//pause();
			}
			else if (contours.size()>0 && max_area<MIN_DETECTABLE_AREA && frameLearnCounter_G == LEARN_FRAME_LIMIT_GREATER_THRES){
				cout << "IN > THRES and condition 2" << endl;
				//Re-initialize the matrix. Very important
				Mat b_hist_back, g_hist_back, r_hist_back;
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;

				background  = nextframe.clone();
				r_hist_back = r_hist_curr.clone();
				g_hist_back = g_hist_curr.clone();
				b_hist_back = b_hist_curr.clone();
				//cout << "Contour area less than MIN_AREA. Updating Background and changing Threshold to: " << *std::max_element(begin(hist_diff_store), end(hist_diff_store)) << endl;
				//CHISQUARE_THRES = *std::max_element(begin(hist_diff_store), end(hist_diff_store));
				cout << "Contour area less than MIN_AREA. Updating Background and changing Threshold to: " << changeCHI << endl;
				CHISQUARE_THRES = changeCHI;
				//hist_diff_store.clear();
				//pause();
			}
			else if (contours.size()>0 && max_area>MIN_DETECTABLE_AREA && max_area<MIN_VEHICLE_AREA && frameLearnCounter_G == LEARN_FRAME_LIMIT_GREATER_THRES){
				cout << "IN > THRES and condition 3" << endl;
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;
				for (int i = 0; i < contours.size(); i++)
				{
					double area = contourArea(contours[i], false);
					if (abs(area - max_area) < 1.0){
						//cout << "rect no " << boundRect[i] << endl;
						boundRect[i] = boundingRect(Mat(contours[i]));
						//cout << "rect no " << boundRect[i] << endl;
						pt1.x = boundRect[i].x;
						pt1.y = boundRect[i].y;
						pt2.x = boundRect[i].x + boundRect[i].width;
						pt2.y = boundRect[i].y + boundRect[i].height;
						rectangle(src, pt1, pt2, Scalar(255), 1);
						cvNamedWindow("Detected Vehicles", CV_WINDOW_NORMAL);
						cvResizeWindow("Detected Vehicles", max_cols, max_rows);
						imshow("Detected Vehicles", src);
						cvMoveWindow("Detected Vehicles", ORIGIN + SHIFT, 350);
						cv::waitKey(30);
					}//end of if
				}//end of for
				hist_diff_store.clear();
				//pause();
			}
			else if (contours.size()>0 && max_area > MIN_VEHICLE_AREA && frameLearnCounter_G == LEARN_FRAME_LIMIT_GREATER_THRES){
				cout << "IN > THRES and condition 4" << endl;
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;
				for (int i = 0; i < contours.size(); i++)
				{
					double area = contourArea(contours[i], false);
					if (abs(area - max_area) < 1.0){
						//cout << "rect no " << boundRect[i] << endl;
						boundRect[i] = boundingRect(Mat(contours[i]));
						//cout << "rect no " << boundRect[i] << endl;
						pt1.x = boundRect[i].x;
						pt1.y = boundRect[i].y;
						pt2.x = boundRect[i].x + boundRect[i].width;
						pt2.y = boundRect[i].y + boundRect[i].height;
						rectangle(src, pt1, pt2, Scalar(255), 1);
						cvNamedWindow("Detected Vehicles", CV_WINDOW_NORMAL);
						cvResizeWindow("Detected Vehicles", max_cols, max_rows);
						imshow("Detected Vehicles", src);
						cvMoveWindow("Detected Vehicles", ORIGIN + SHIFT, 350);
						cout << "Vehicle height is:" <<pt1.y<< endl;
						if (pt1.y < MIN_VEHICLE_HEIGHT){
							cout << "Bad rectangle found. Updating Background" << endl;
							//Re-initialize the matrix. Very important
							Mat b_hist_back, g_hist_back, r_hist_back;
							frameLearnCounter_G = 0;
							frameLearnCounter_L = 0;

							background = nextframe.clone();
							r_hist_back = r_hist_curr.clone();
							g_hist_back = g_hist_curr.clone();
							b_hist_back = b_hist_curr.clone();
							pause();
						}
						cv::waitKey(30);
					}//end of if
				}//end of for
				hist_diff_store.clear();
				//pause();
			}
			
		}//end of if (changeCHI>CHISQUARE_THRES)

		outFile << r_changeCHI << endl;
		int x=cvWaitKey(30);
		if (x == 32)
			pause();
	}//end of while
	int a = cvWaitKey(100000);
	return 0;
}//end of main