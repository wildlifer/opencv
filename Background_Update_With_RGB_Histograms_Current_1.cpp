#include "helper_cv.h"
int frameLearnCounter_L = 0;
int frameLearnCounter_G = 0;
int CHISQUARE_THRES = 0;

int main(int argc, char** argv)
{
	vector<Mat> initFrames;
	//No of Vehicles
	int no_of_vehicles = 0;
	vector<vector<double>> vehicles;
	//File
	ofstream outFile;
	//Video
	VideoCapture cap("videos/4.5.mp4");
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
	r_hist_back=hist[0].clone();
	g_hist_back=hist[1].clone();
	b_hist_back=hist[2].clone();
	//Create temporary images
	CvSize sz = cvGetSize(new IplImage(background));

	Mat agray(background);
	Mat bgray(background);
	Mat diff(background);

	//store previous images
	vector<Mat> matStore(LEARN_FRAME_LIMIT_LESS_THRES + 1);
	Mat rep_background; //replaceable backgrond
	Mat b_hist_back_rep, g_hist_back_rep, r_hist_back_rep;
	vector<int> hist_diff_store_less(LEARN_FRAME_LIMIT_LESS_THRES, 0);
	vector<int> hist_diff_store_greater(LEARN_FRAME_LIMIT_GREATER_THRES, 0);
	while (1) {
		//clear planes
		bgr_planes_curr_frame.clear();
		//new frame
		cap >> nextframe;
		if (nextframe.empty()) break;
		//get region of interest
		if (ROI_ENABLED){
			nextframe = nextframe(roi);
		}
		vector<Mat>hist = getNormalizedHist(nextframe, histSize, range, uniform, accumulate);
		r_hist_curr = hist[0].clone();
		g_hist_curr = hist[1].clone();
		b_hist_curr = hist[2].clone();

		//Show Original Video  - IplImage model
		cvtColor(nextframe, bgray, CV_BGR2GRAY);
		medianBlur(bgray, bgray, APERTURE);
		createNMoveWindow("Original Video", max_cols, max_rows, nextframe, ORIGIN + SHIFT, 0);

		//Show current background
		cvtColor(background, agray, CV_BGR2GRAY);
		medianBlur(agray, agray, APERTURE);
		createNMoveWindow("Current background", max_cols, max_rows, background, ORIGIN + SHIFT + WINDISTANCE + max_cols, 0);

		//Get the difference between background and current frame
		absdiff(agray, bgray, diff);
		medianBlur(diff, diff, APERTURE);
		IplImage *I = new IplImage(diff);
		cvThreshold(I, I, 25, 255, CV_THRESH_BINARY);
		Mat m(I);
		diff = m.clone();
		createNMoveWindow("Diff Frame", max_cols, max_rows, diff, ORIGIN + SHIFT + WINDISTANCE + max_cols, 350);	

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

		double changeCHI = B_WEIGHT*b_changeCHI + R_WEIGHT*r_changeCHI + G_WEIGHT*g_changeCHI;

		cout << "Change in Chi is " << (changeCHI) << endl;
		
		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
		int i = 1;

		Mat src=diff.clone();
		Mat original=bgray.clone();
		vector<vector<Point>> contours;
		Rect boundRec;
		vector<Vec4i> hierarchy;
		//Canny(src, canny_output, thresh, thresh * 2, 3);
		//CV_RETR_EXTERNAL is important here. It helps to find only outer contours which is what we need
		findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		Mat noise_less_frame = drawContoursNDeNoise2(diff, contours, MIN_DETECTABLE_AREA);
		//createNMoveWindow("Noise Less", max_cols, max_rows, noise_less_frame, ORIGIN, 350);
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
		//Threshold comparison begins here
		if (changeCHI <= CHISQUARE_THRES){
			//Re-initialize the matrix. Very important
			Mat b_hist_back, g_hist_back, r_hist_back;

			if (frameLearnCounter_L < LEARN_FRAME_LIMIT_LESS_THRES){
				//cout << "In < THRES and attempting insert in hist_diff_store_less with  counter as: " << frameLearnCounter_L << endl;
				hist_diff_store_less[frameLearnCounter_L] = (int)changeCHI;
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

				if (contours.size() == 0){
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
					cout << "Updating Background as detected area is less than MAX_AREA in < THRES" << endl;
					cout << "*****************************************" << endl;
					CHISQUARE_THRES = *std::max_element(begin(hist_diff_store_less), end(hist_diff_store_less));
					cout << "Updating CHISQUARE_THRES to: " << changeCHI << endl;
					cout << "*****************************************" << endl; 
					drawContours(src, contours, max_area);
					//pause();
				}
				else if (contours.size() > 0 && max_area > MIN_DETECTABLE_AREA && max_area < MIN_VEHICLE_AREA){
					cout << "IN < THRES and condition 3 with " << contours.size() << " contours" << endl;
					pause();
				}
				else if (contours.size() > 0 && max_area >= MIN_VEHICLE_AREA){
					cout << "IN < THRES and condition 4 with " << contours.size() << " contours as threshold = "<< CHISQUARE_THRES<< endl;
					/*for (int i = 0; i < contours.size(); i++){
						double area = contourArea(contours[i], false);
						//cout << "rect no " << boundRect[i] << endl;
						boundRect[i] = boundingRect(Mat(contours[i]));
						pt1.x = boundRect[i].x;
						pt1.y = boundRect[i].y;
						pt2.x = boundRect[i].x + boundRect[i].width;
						pt2.y = boundRect[i].y + boundRect[i].height;
						if (area>= MIN_VEHICLE_AREA && pt2.y>=MIN_VEHICLE_HEIGHT){	
							cout << "Vehicle height is:" << pt2.y << endl;
							cout << "Vehicle width is:" << pt2.x << endl;
							cout << "Vehicle area is:" << area << endl;
							rectangle(src, pt1, pt2, Scalar(255), 1);
							createNMoveWindow("Detected Vehicles", max_cols, max_rows, src, ORIGIN + SHIFT, 350);
							if (pt2.y < MIN_VEHICLE_HEIGHT || contours.size() > MAX_CONTOURS){
								cout << "Bad rectangle found or contours exceed the allowed limit. Updating Background with replacement" << endl;
								frameLearnCounter_G = 0;
								frameLearnCounter_L = 0;
								background = rep_background.clone();
								r_hist_back = r_hist_back_rep.clone();
								g_hist_back = g_hist_back_rep.clone();
								b_hist_back = b_hist_back_rep.clone();
								pause();
							}
						
						}//end of if
					}*/

					vector<double>vehicle(4, 0);
					no_of_vehicles = 0;
					for (int i = 0; i < contours.size(); i++)
					{
						double area = contourArea(contours[i], false);
						boundRect[i] = boundingRect(Mat(contours[i]));
						vehicle[0] = pt1.x = boundRect[i].x; //x 
						vehicle[1] = pt1.y = boundRect[i].y; //y
						vehicle[2] = pt2.x = boundRect[i].x + boundRect[i].width;//width
						vehicle[3] = pt2.y = boundRect[i].y + boundRect[i].height;//height
						cout << "Vehicle height is:" << boundRect[i].height << endl;
						cout << "Vehicle width is:" << boundRect[i].width << endl;
						cout << "Vehicle area is:" << area << endl;
						cout << "----------------------------------" << endl;
						rectangle(src, pt1, pt2, Scalar(255), 1);
						createNMoveWindow("Detected Vehicles", max_cols, max_rows, src, ORIGIN + SHIFT, 350);

						if (area >= MIN_DETECTABLE_AREA && boundRect[i].height >= MIN_VEHICLE_HEIGHT && boundRect[i].width >= MIN_VEHICLE_WIDTH ||
							area >= MIN_VEHICLE_AREA && boundRect[i].height <= MIN_VEHICLE_HEIGHT && boundRect[i].width >= MIN_VEHICLE_WIDTH ||
							area >= MIN_VEHICLE_AREA && boundRect[i].height >= MIN_VEHICLE_HEIGHT && boundRect[i].width <= MIN_VEHICLE_WIDTH){
							no_of_vehicles++;
						}
					}
					cout << "No of vehicles are: " << no_of_vehicles << endl;
					//CHISQUARE_THRES = *std::max_element(begin(hist_diff_store_less), end(hist_diff_store_less));
					//CHISQUARE_THRES = CHISQUARE_THRES - 0.25*CHISQUARE_THRES;
					//cout << "Updating CHISQUARE_THRES to: " << CHISQUARE_THRES << " in < THRES condi 4" << endl;
					cout << "*****************************************" << endl;
					pause();
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
					cout << "Updating Background as contour area is less than Min detectable area in > THRES. Area is: " << max_area << endl;
					cout << "*****************************************" << endl;
					CHISQUARE_THRES = *std::min_element(begin(hist_diff_store_greater), end(hist_diff_store_greater));
					cout << "Updating CHISQUARE_THRES to: " << CHISQUARE_THRES << endl;
					cout << "*****************************************" << endl;
					//pause();
				}
				else if (contours.size() > 0 && max_area > MIN_DETECTABLE_AREA && max_area < MIN_VEHICLE_AREA){
					cout << "IN > THRES and condition 3 with " << contours.size() << " contours and area: " << max_area <<endl;
					drawContours(src, contours, max_area);
					vector<double>vehicle(4, 0);
					no_of_vehicles = 0;
					for (int i = 0; i < contours.size(); i++)
					{
						double area = contourArea(contours[i], false);
						boundRect[i] = boundingRect(Mat(contours[i]));
						vehicle[0] = pt1.x = boundRect[i].x; //x 
						vehicle[1] = pt1.y = boundRect[i].y; //y
						vehicle[2] = pt2.x = boundRect[i].x + boundRect[i].width;//width
						vehicle[3] = pt2.y = boundRect[i].y + boundRect[i].height;//height
						cout << "Vehicle height is:" << boundRect[i].height << endl;
						cout << "Vehicle width is:" << boundRect[i].width << endl;
						cout << "Vehicle area is:" << area << endl;
						cout << "----------------------------------"<< endl;
						rectangle(src, pt1, pt2, Scalar(255), 1);
						createNMoveWindow("Detected Vehicles", max_cols, max_rows, src, ORIGIN + SHIFT, 350);

						if (area >= MIN_DETECTABLE_AREA && boundRect[i].height >= MIN_VEHICLE_HEIGHT && boundRect[i].width >= MIN_VEHICLE_WIDTH ||
							area >= MIN_VEHICLE_AREA && boundRect[i].height <= MIN_VEHICLE_HEIGHT && boundRect[i].width >= MIN_VEHICLE_WIDTH ||
							area >= MIN_VEHICLE_AREA && boundRect[i].height >= MIN_VEHICLE_HEIGHT && boundRect[i].width <= MIN_VEHICLE_WIDTH){
							no_of_vehicles++;
						}
					}
					cout << "No of vehicles are: " << no_of_vehicles << endl;

					pause();
				}
				else if (contours.size() > 0 && max_area > MIN_VEHICLE_AREA){
					cout << "IN > THRES and condition 4 with " << contours.size() << " contours" << endl;
					vector<double>vehicle(4, 0);
					no_of_vehicles = 0;
					for (int i = 0; i < contours.size(); i++)
					{
						double area = contourArea(contours[i], false);
						boundRect[i] = boundingRect(Mat(contours[i]));
						vehicle[0] = pt1.x = boundRect[i].x; //x 
						vehicle[1] = pt1.y = boundRect[i].y; //y
						vehicle[2] = pt2.x = boundRect[i].x + boundRect[i].width;//width
						vehicle[3] = pt2.y = boundRect[i].y + boundRect[i].height;//height
						/*cout << "Vehicle height is:" << boundRect[i].height << endl;
						cout << "Vehicle width is:" << boundRect[i].width << endl;
						cout << "Vehicle area is:" << area << endl;*/
						rectangle(src, pt1, pt2, Scalar(255), 1);
						createNMoveWindow("Detected Vehicles", max_cols, max_rows, src, ORIGIN + SHIFT, 350);

						if (area >= MIN_VEHICLE_AREA && boundRect[i].height >= MIN_VEHICLE_HEIGHT && boundRect[i].width >= MIN_VEHICLE_WIDTH  ||
							area >= MIN_VEHICLE_AREA && boundRect[i].height <= MIN_VEHICLE_HEIGHT && boundRect[i].width >= MIN_VEHICLE_WIDTH  ||
							area >= MIN_VEHICLE_AREA && boundRect[i].height >= MIN_VEHICLE_HEIGHT && boundRect[i].width <= MIN_VEHICLE_WIDTH){
							no_of_vehicles++;		
						}
					}
					cout << "No of vehicles are: " << no_of_vehicles << endl;
					
					for (int i = 0; i < contours.size(); i++)
					{
						double area = contourArea(contours[i], false);
						boundRect[i] = boundingRect(Mat(contours[i]));
						vehicle[0] = pt1.x = boundRect[i].x; //x 
						vehicle[1] = pt1.y = boundRect[i].y; //y
						vehicle[2] = pt2.x = boundRect[i].x + boundRect[i].width;//width
						vehicle[3] = pt2.y = boundRect[i].y + boundRect[i].height;//height
					
						if (area >= MIN_VEHICLE_AREA && boundRect[i].height >= MIN_VEHICLE_HEIGHT && boundRect[i].width >= MIN_VEHICLE_WIDTH){
							if (vehicles.empty()){//Record the parameters of the vehicle
								//cout << "First Vehicle counted" << endl;
								//pause();
								vehicles.push_back(vehicle);
							}
							else{
								/*for (int i = 0; i < vehicles.size; i++){
									vector<double> vec = vehicles.at(i);
									if (abs(vec[1] - vehicle[1]) <= MAX_Y_TOLERANCE && abs(vec[3] - vehicle[3]) <= MAX_HEIGHT_TOLERANCE){

									}
									else{

									}
								}*/
							}
							
							if (pt2.y < MIN_VEHICLE_HEIGHT){
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
					}
					//pause();
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