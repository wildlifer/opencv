#include "helper_cv.h"
int frameLearnCounter_L = 0;
int frameLearnCounter_G = 0;
int CHISQUARE_THRES = 0;

int main(int argc, char** argv)
{
	vector<Mat> initFrames;
	ofstream outFile;
	VideoCapture cap("videos/4.1.mp4");
	int threshold = getBackgroundThreshold(cap, INITIAL_LEARN_FRAME_LIMIT);
	CHISQUARE_THRES = threshold;;
	Mat background, hsv_background, gray_background;
	Mat nextframe, hsv_nextframe, gray_nextframe;
	Mat diff_frame;
	outFile.open("data/VehicleCassifier/changeInMetric.txt", 'w');

	/// Separate the image in 3 places ( B, G and R )/
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
	
	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	/// Normalize the result to [ 0, histImage.rows ]
	
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
	int ii = 0;
	//store previous images
	vector<Mat> matStore(LEARN_FRAME_LIMIT_LESS_THRES + 1);
	Mat rep_background; //replaceable backgrond
	Mat b_hist_back_rep, g_hist_back_rep, r_hist_back_rep;
	vector<int> hist_diff_store_less(LEARN_FRAME_LIMIT_LESS_THRES,0);
	vector<int> hist_diff_store_greater(LEARN_FRAME_LIMIT_GREATER_THRES,0);
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
		
		//Show Original Video  - IplImage model
		frame = new IplImage(nextframe);
		cvCvtColor(frame, bgray, CV_BGR2GRAY);
		cvSmooth(bgray, bgray, CV_MEDIAN);
		createNMoveWindow("Original Video", max_cols, max_rows, frame, ORIGIN + SHIFT, 0);

		//Show current background
		oldFrame = new IplImage(background);
		cvCvtColor(oldFrame, agray, CV_BGR2GRAY);
		cvSmooth(agray, agray, CV_MEDIAN);
		createNMoveWindow("Current background", max_cols, max_rows, oldFrame, ORIGIN + SHIFT + WINDISTANCE + max_cols, 0);

		//Get the difference between background and current frame
		cvAbsDiff(agray, bgray, diff);
		cvSmooth(diff, diff, CV_MEDIAN);
		cvThreshold(diff, diff, 25, 255, CV_THRESH_BINARY);

		////floodfill here
		//Mat diff_clone(diff);
		//Mat diff_clone1(diff);
		//floodFill(diff_clone1, cv::Point2i(0, 0), cv::Scalar(0,0,0),0,20,20,4);
	
		//for (int i = 0; i<diff_clone.rows*diff_clone.cols; i++)
		//{
		//	if (diff_clone1.data[i] == 1){
		//		diff_clone.data[i] = 0;
		//		cout << "Repainting to 0" << endl;
		//	}
		//}
		//imshow("Diff Clone 1",diff_clone1);
		//imshow("Holes", diff_clone);
		//cvWaitKey(30);
		//pause();

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
		vector<vector<Point>> contours;
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
			//Re-initialize the matrix. Very important
			Mat b_hist_back, g_hist_back, r_hist_back;

			if (frameLearnCounter_L < LEARN_FRAME_LIMIT_LESS_THRES){
				//cout << "In < THRES and attempting insert in hist_diff_store_less with  counter as: " << frameLearnCounter_L << endl;
				hist_diff_store_less[frameLearnCounter_L]=(int)changeCHI;
				//hist_diff_store_less.push_back(changeCHI);
				//cout << "Pushing in " << hist_diff_store_less[frameLearnCounter_G] << endl;
				frameLearnCounter_L++;
			}
			else if (frameLearnCounter_L > LEARN_FRAME_LIMIT_LESS_THRES){
				frameLearnCounter_L = 0;
				frameLearnCounter_G = 0;
				//cout << "Attempting clear of hist_diff_store_less" << endl;
				fill(hist_diff_store_less.begin(), hist_diff_store_less.end(), 0);
			}
			//Keep previous copies of similar images
			//if (frameLearnCounter<LEARN_FRAME_LIMIT)
			//nextframe.copyTo(matStore[frameLearnCounter]);
			if (frameLearnCounter_L == LEARN_FRAME_LIMIT_LESS_THRES){
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;
				
				if (contours.size()==0){
					cout << "IN < THRES and condition 1" << endl;
					background = nextframe.clone();
					r_hist_back = r_hist_curr.clone();
					g_hist_back = g_hist_curr.clone();
					b_hist_back = b_hist_curr.clone();
					//store for replacement
					rep_background = nextframe.clone();
					r_hist_back_rep = r_hist_curr.clone();
					g_hist_back_rep = g_hist_curr.clone();
					b_hist_back_rep = b_hist_curr.clone();
					//cout << "Updating Background and changing Threshold to: " << *std::max_element(begin(hist_diff_store), end(hist_diff_store)) << endl;
					//CHISQUARE_THRES = *std::max_element(begin(hist_diff_store), end(hist_diff_store));
					cout << "Updating Background as no contours found in < THRES" << endl;
					cout << "*****************************************" << endl;
					CHISQUARE_THRES = *std::max_element(begin(hist_diff_store_less), end(hist_diff_store_less));
					cout << "Updating CHISQUARE_THRES to: " << changeCHI << endl;
					cout << "*****************************************" << endl;	
					
					//pause();
				}
				else if (contours.size() > 0 && max_area < MIN_DETECTABLE_AREA){
					cout << "IN < THRES and condition 2 with " << contours.size() << " contours" << endl;
					//matStore.clear();
					//copy current frame to the background
					background = nextframe.clone();
					
					r_hist_back = r_hist_curr.clone();
					
					g_hist_back = g_hist_curr.clone();
					
					b_hist_back = b_hist_curr.clone();
					
					//store for replacement
					rep_background = nextframe.clone();
					r_hist_back_rep = r_hist_curr.clone();
					g_hist_back_rep = g_hist_curr.clone();
					b_hist_back_rep = b_hist_curr.clone();
					CHISQUARE_THRES = *std::max_element(begin(hist_diff_store_less), end(hist_diff_store_less));
					cout << "Contour area less than MIN_AREA. Updating Background and changing Threshold to: " << CHISQUARE_THRES << endl;
					drawContours(src, contours, max_area);
					//pause();
				}
				else if (contours.size() > 0 && max_area > MIN_DETECTABLE_AREA && max_area < MIN_VEHICLE_AREA){
					cout << "IN < THRES and condition 3 with " << contours.size() << " contours" << endl;
					//pause();
				}
				else if (contours.size() > 0 && max_area > MIN_VEHICLE_AREA){
					cout << "IN < THRES and condition 4 with " << contours.size() << " contours" << endl;
					for (int i = 0; i < contours.size(); i++){
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
							createNMoveWindow("Detected Vehicles", max_cols, max_rows, src, ORIGIN + SHIFT, 350);
							cout << "Vehicle height:" << pt2.y << " and Vehicle width: " << pt2.x << " with " << contours.size() << " contours"<< endl;
							cout << "Vehicle area" << area << endl;
							if (pt2.y < MIN_VEHICLE_HEIGHT || contours.size() > MAX_CONTOURS){
								cout << "Bad rectangle found or contours exceed the allowed limit. Updating Background with replacement" << endl;
								frameLearnCounter_G = 0;
								frameLearnCounter_L = 0;

								/*background = rep_background.clone();
								r_hist_back = r_hist_back_rep.clone();
								g_hist_back = g_hist_back_rep.clone();
								b_hist_back = b_hist_back_rep.clone();
								pause();*/
							}
							cv::waitKey(30);
						}//end of if
					}
					//pause();
				}
				fill(hist_diff_store_less.begin(), hist_diff_store_less.end(), 0);
			}		
		}//end of if(changeCHI < CHISQUARE_THRES)
		else if (changeCHI>CHISQUARE_THRES){
			//cout << "Above Threshold with frame counter: " << frameLearnCounter_G << endl;
			if (frameLearnCounter_G < LEARN_FRAME_LIMIT_GREATER_THRES){
				//cout << "In > THRES and attempting insert in hist_diff_store_greater" << endl;
				hist_diff_store_greater[frameLearnCounter_G] = (int)changeCHI;
				//cout << "Pushing in " << hist_diff_store_greater[frameLearnCounter_G] << endl;
				frameLearnCounter_G++;
			}
			else if (frameLearnCounter_G > LEARN_FRAME_LIMIT_GREATER_THRES){
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;
				//cout << "Attempting clear of hist_diff_store_greater" << endl;
				fill(hist_diff_store_greater.begin(), hist_diff_store_greater.end(), 0);
			}
			//matStore.clear();
			if (frameLearnCounter_G == LEARN_FRAME_LIMIT_GREATER_THRES){
				frameLearnCounter_G = 0;
				frameLearnCounter_L = 0;
				
				if (contours.size() == 0){
					cout << "IN > THRES and condition 1" << endl;
					//Re-initialize the matrix. Very important
					Mat b_hist_back, g_hist_back, r_hist_back;
					//copy current frame to the background
					background = nextframe.clone();
					r_hist_back = r_hist_curr.clone();
					g_hist_back = g_hist_curr.clone();
					b_hist_back = b_hist_curr.clone();
					cout << "Updating Background as no contours found in > THRES" << endl;
					cout << "*****************************************" << endl;
					CHISQUARE_THRES = *std::min_element(begin(hist_diff_store_greater), end(hist_diff_store_greater));
					cout << "Updating CHISQUARE_THRES to: " << CHISQUARE_THRES << endl;
					cout << "*****************************************" << endl;
					
					//pause();
				}
				else if (contours.size() > 0 && max_area < MIN_DETECTABLE_AREA){//Very likely condition
					cout << "IN > THRES and condition 2 with " << contours.size() << " contours" << endl;
					//Re-initialize the matrix. Very important
					Mat b_hist_back, g_hist_back, r_hist_back;
					background = nextframe.clone();
					r_hist_back = r_hist_curr.clone();
					g_hist_back = g_hist_curr.clone();
					b_hist_back = b_hist_curr.clone();
					//cout << "Contour area less than MIN_AREA. Updating Background and changing Threshold to: " << *std::max_element(begin(hist_diff_store), end(hist_diff_store)) << endl;
					cout << "Updating Background as contour area is less than Min detectable area. Area is: "<< max_area<< endl;
					cout << "*****************************************" << endl;
					CHISQUARE_THRES = *std::min_element(begin(hist_diff_store_greater), end(hist_diff_store_greater));
					cout << "Updating CHISQUARE_THRES to: " << CHISQUARE_THRES << endl;
					cout << "*****************************************" << endl; 
					
					//pause();
				}
				else if (contours.size() > 0 && max_area > MIN_DETECTABLE_AREA && max_area < MIN_VEHICLE_AREA){
					cout << "IN > THRES and condition 3 with " << contours.size()<< " contours" <<endl;
					drawContours( src, contours, max_area);
					//pause();
				}
				else if (contours.size() > 0 && max_area > MIN_VEHICLE_AREA){
					cout << "IN > THRES and condition 4 with " << contours.size() << " contours" << endl;
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
							createNMoveWindow("Detected Vehicles", max_cols, max_rows, src, ORIGIN + SHIFT, 350);
							cout << "Vehicle height is:" << pt1.y << endl;
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
								//pause();
							}
							cv::waitKey(30);
						}//end of if
					}//end of for
					fill(hist_diff_store_greater.begin(), hist_diff_store_greater.end(), 0);
					if (contours.size()>MAX_CONTOURS){
						cout << "Contours exceed max allowed " << endl;
						//pause();
					}
				}
			}
			
		}//end of if (changeCHI>CHISQUARE_THRES)

		outFile << CHISQUARE_THRES << endl;
		int x = cvWaitKey(30);
		if (x == 32)
			pause();
	}//end of while
	int a = cvWaitKey(100000);
	return 0;
}//end of main