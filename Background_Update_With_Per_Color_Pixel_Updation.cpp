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
	VideoCapture cap(video);
	vector<Mat> training(TRAINING_FRAMES);
	vector<Mat> no_contour_frame_vec(NO_CONTOUR_FRAME_COUNT);
	Mat background, gray_background, backup_background;;
	Mat nextframe, gray_nextframe;
	Mat diff_frame;
	outFile.open("data/VehicleCassifier/changeInMetric.txt", 'w');
	cap >> background;
	cap >> background;
	for (int i = 0; i < TRAINING_FRAMES; i++){
		cap >> background;
		training[i]=background.clone();
	}

	double THRES = getBackgroundThreshold(training);
	outFile << std::fixed << std::setprecision(2) << THRES; outFile << endl;

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
	createNMoveWindow("Detected objects", max_cols, max_rows, background, ORIGIN + SHIFT, 350);
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
		//if (frame_no < 2385){
			//background = nextframe.clone();
			//continue;
		//}
		
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
		imshow("Original Video",  nextframe);

		//Show current background
		cvtColor(background, gray_background, CV_BGR2GRAY);
		medianBlur(gray_background, gray_background, APERTURE);
		imshow("Current background",  background);

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
		//Canny(src, canny_output, thresh, thresh * 2, 3);
		//CV_RETR_EXTERNAL is important here. It helps to find only outer contours which is what we need
		findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		double max_area_diff_frame = 300;
		Point p,q;
		int wid, hi = 0;
		/***************Check for max area contour and fill the max_area with whites***************/
		for (int i = 0; i < contours.size(); i++){
			int a = 0;
			if (contourArea(contours[i], false) > max_area_diff_frame){
				//max_area_diff_frame = contourArea(contours[i]);
				a = 1;
				boundRec = boundingRect((contours[i]));
				p.x = boundRec.x;
				p.y = boundRec.y;
				q.x = boundRec.x + boundRec.width;
				q.y = boundRec.y + boundRec.height;
			}
			if (a == 1){
				for (int i = p.x; i < q.x; i++){
					for (int j = p.y; j < q.y; j++){
						diff.at<uchar>(Point(i, j)) = 255;
					}
				}
			}
		}
		
	
		//imshow("Diff Frame After updation", diff);
		imshow("Diff Frame", diff);
		//cvWaitKey(30);

		/***************Check for max area contour and fill the max_area with whites***************/

		/*Denoise the frame and find contours again for extra precision*/
		Mat noise_less_frame = drawContoursNDeNoise2(diff, contours, MIN_DETECTABLE_AREA);
		findContours(noise_less_frame, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		src = noise_less_frame.clone();

		/*Use the diff frame without denoising*/
		//Mat noise_less_frame = src.clone();

		vector<Rect> boundRect(contours.size());
		//Point pt1, pt2;
		int vehicle_width, vehicle_height;
		double max_area = 0;
		Point max_area_point;
		bool has_potential_vehicle = false;
		no_of_vehicles_in_frame = 0;
		bool copy_frame_as_background = true;
		bool check_for_min_contour = true;
		bool possible_merge = false;

		/********Check for possible merge of vehicles*******/
		for (int i = 0; i < vehicles-1; i++){
			/*cout << "vehicles_detected[i][4]= " << vehicles_detected[i][4] << endl;
			cout << "vehicles_detected[i+1][0]= " << vehicles_detected[i + 1][0] << endl;
			cout << "vehicles_detected[i][0]= " << vehicles_detected[i][0] << endl;
			cout << "vehicles_detected[i+1][4]= " << vehicles_detected[i + 1][4] << endl;
			*/
			if (abs(vehicles_detected[i][4] - vehicles_detected[i+1][0]) < MERGE_THRES){
				cout << "vehicles_detected[i][4]= " << vehicles_detected[i][4]<< endl;
				cout << "vehicles_detected[i+1][0]= " << vehicles_detected[i+1][0] << endl;
				possible_merge = true;
			}
			else if (abs(vehicles_detected[i][0] - vehicles_detected[i+1][4]) < MERGE_THRES){
				cout << "vehicles_detected[i][0]= " << vehicles_detected[i][0] << endl;
				cout << "vehicles_detected[i+1][4]= " << vehicles_detected[i + 1][4] << endl;
				possible_merge = true;
			}
		}
		/********Check for possible merge of vehicles*******/

		if (possible_merge){
			cout << "################Possible MERGE ahead###################" << endl;
			pause();
		}
		
		//cout << "No of contours are :" << contours.size() << endl;
		/**************Contour check**************/
		for (int i = 0; i < contours.size(); i++){
			Point pt1, pt2, pt3, pt4;
			boundRect[i] = boundingRect(Mat(contours[i]));
			pt1.x = boundRect[i].x;
			pt1.y = boundRect[i].y;
			pt2.x = pt1.x + boundRect[i].width;
			pt2.y = pt1.y;
			pt3.x = pt1.x;
			pt3.y = pt1.y + boundRect[i].height;
			pt4.x = pt2.x;
			pt4.y = pt2.y + boundRect[i].height;
			vehicle_width = boundRect[i].width;
			vehicle_height = boundRect[i].height;

			bool isNormal = true;
			/*************Check for abnormal contours*****************/
			double width_to_height_ratio = vehicle_width / vehicle_height;
			if ((vehicle_height < 15 && width_to_height_ratio > MAX_WIDTH_TO_HEIGHT_RATIO) ||
				(vehicle_height > 15 && width_to_height_ratio < MIN_WIDTH_TO_HEIGHT_RATIO) ||
				(vehicle_width < MIN_VEHICLE_WIDTH) ){
				//cout << "Abnormal contour detected" << endl;
				isNormal = false;
				//pause();
				//continue;
			}
			/*************Check for abnormal contours****************/

			if (contourArea(contours[i], false) > MIN_DETECTABLE_AREA && check_for_min_contour ==true){
				 check_for_min_contour = false;
				 copy_frame_as_background = false;
				 has_potential_vehicle = true;
			}

			/************Check for possible vehicles*********/
			if (isNormal && contourArea(contours[i], false) > MIN_DETECTABLE_AREA &&//1
				vehicle_width > MIN_VEHICLE_WIDTH &&//2
				vehicle_height > MIN_VEHICLE_HEIGHT &&//3
				no_of_vehicles_in_frame <= MAX_VEHICLES_IN_FRAME){//4

				/***************Check for max area contour***************/
				if (contourArea(contours[i], false) > max_area){
					max_area = contourArea(contours[i]);
					boundRec = boundingRect((contours[i]));
					max_area_point.x = boundRec.x;
					max_area_point.y = boundRec.y;
				}
				/***************Check for max area contour***************/
				
				/****Find number of vechicles already detected*********/			
				//for (int i = 0; i < MAX_VEHICLES_IN_FRAME; i++){
					//if (vehicles_detected[i][0]!=0){
						//vehicles++;
			//		}
				//}
				/****Find number of vechicles already detected*********/

				/*check if the vehicle has already been detected*/
				bool vec_check = false;
				for (int i = 0; i < MAX_VEHICLES_IN_FRAME; i++){

					if ( (((abs(vehicles_detected[i][1] - pt1.y)  < MAX_Y_TOLERANCE) &&
						 (abs(vehicles_detected[i][3] - vehicle_height) < MAX_Y_TOLERANCE)) //1:Check for similar height and y coordinates
						||//else check for similar width and height
						 ((abs(vehicles_detected[i][2] - vehicle_width) < MAX_X_TOLERANCE) &&
						  (abs(vehicles_detected[i][3] - vehicle_height) < MAX_Y_TOLERANCE))	
						||//else check for width and y coordinates
						 ((abs(vehicles_detected[i][2] - vehicle_width) < MAX_X_TOLERANCE) &&
						 (abs(vehicles_detected[i][1] - pt1.y) < MAX_Y_TOLERANCE))) 
						&&//2:Check for 
						//(abs(vehicles_detected[i][0] - pt1.x) < (vehicle_width + MAX_X_TOLERANCE))
						(abs(vehicles_detected[i][0] - pt1.x) < (MAX_X_TOLERANCE+20))//Newly added
					   )
					{
						vec_check = true;
						//cout << "***************Vehicle no " << i+1 << " already exists****************" << endl;
						/*cout << "Detected Vehicle width: " << vehicle_width << " and Stored vehicle width: " << vehicles_detected[i][2] << endl;
						cout << "Detected Vehicle height: " << vehicle_height << " and Stored vehicle height: " << vehicles_detected[i][3] << endl;
						cout << "Detected Vehicle x: " << pt1.x << " and Stored vehicle x: " << vehicles_detected[i][0] << endl;
						cout << "Detected Vehicle y: " << pt1.y << " and Stored vehicle y: " << vehicles_detected[i][1] << endl;
						cout << "X difference is " << (abs(vehicles_detected[i][0] - pt1.x))  << endl;
						cout << "*****************************************************" << endl;*/
						
						/*Find the direction of the vehicle*/
						if (pt1.x >= vehicles_detected[i][0] && pt2.x >= vehicles_detected[i][4]){
							
							cout << "Vehicle travelling RIGHT >>>>>>>>>>>>>>> at " << pt1.x << ","<< pt2.x<< endl;
							/*Direction update*/
							vehicles_detected[i][6] = 2;
							/*Exit check*/
							if (pt2.x > (frame_width - (5)) && pt2.x < frame_width){
								cout << ">>>>>>>>>>Vehicle Exiting RIGHT>>>>>>>>>>" << endl;
								/*Now remove vehicle from the memory*/
								for (int j = 0; j < NO_OF_CUES; j++){
									vehicles_detected[i][j] = 0;
								}
								cout << "Vehicle removed from memory" << endl;
								--vehicles;
								cout << "Vehicle left in memory: " << vehicles << endl;
								//pause();
							}
						}
						else if (pt1.x <= vehicles_detected[i][0] && pt2.x <= vehicles_detected[i][4]){
						/*	cout << "pt1.x = " << pt1.x << endl;
							cout << "vehicles_detected[i][0] = " << vehicles_detected[i][0] << endl;
							cout << "pt2.x = " << pt2.x << endl;
							cout << "vehicles_detected[i][4] = " << vehicles_detected[i][4] << endl;
							*/
							cout << "Vehicle travelling LEFT <<<<<<<<<<<<<<<<<<at " << pt1.x << "," << pt2.x << endl;
							/*Direction update*/
							vehicles_detected[i][6] = 1;
							/*Exit check*/
							if (pt1.x > 0 && pt1.x < 5){
								cout << "<<<<<<<<<<<Vehicle Exiting LEFT<<<<<<<<<<<<" << endl;
								/*Now remove vehicle from the memory*/
								for (int j = 0; j < NO_OF_CUES; j++){
									vehicles_detected[i][j] = 0;
								}
								cout << "Vehicle removed from memory" << endl;
								--vehicles;
								cout << "Vehicle left in memory: " << vehicles << endl;
								//pause();
							}
						}
						//update the x1 and x2 coordinates of the vehicle
						vehicles_detected[i][0] = pt1.x;
						vehicles_detected[i][4] = pt2.x;
						
						//pause();
						break;
					}
					else if ((pt1.x >= vehicles_detected[i][0] + 10 && pt1.x <= vehicles_detected[i][0]+ vehicles_detected[i][2] + 10) &&
						(pt2.x >= vehicles_detected[i][0] + 10 && pt2.x <= vehicles_detected[i][0] + vehicles_detected[i][2] + 10))
					{ //Check for split contours
						cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << endl;
						cout << "------Split vehicle detected-------" << endl;
						cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$" << endl;
						vec_check = true;
						pause();
						break;
					}
				}//end of for

				/*********************check for a new vehicle********************/
				if (!possible_merge && !vec_check && pt1.x > 2 * MIN_VEHICLE_WIDTH && pt1.x < (frame_width - 3 * MIN_VEHICLE_WIDTH) &&
					pt2.x < (frame_width - 3*MIN_VEHICLE_WIDTH) )
				{
					Mat a = unblur_gray_nextframe(Rect(pt1.x-25, pt1.y-25, vehicle_width+25, vehicle_height + 45));
					/*Mat detected_edges;
					Canny(a, detected_edges, 1, 3, 3);
					Mat dst;// = Scalar::all(0);
					a.copyTo(dst, detected_edges);
					imshow("Edges", detected_edges);*/

					bool circle_found = checkCircles(a);
					if (circle_found){
						cout << "-----------------Circle found------------------" << endl;
						pause();
					}
					/*Increment total vehicle count*/
					total_vechicles++;
					/*Increment total vehicle count*/

					/*Display the coordinates and features of the newly detected vehicle*/
					cout << "+++++++++++++++++++++++++++++++++++++" << endl;
					cout << "New Vehicle detected with i as " << i << " at->>("<< pt1.x << ","<< pt1.y << "):(" << pt2.x <<","<<pt2.y<<")"<<  endl;
					/*cout << "Detected Vehicle width: " << vehicle_width << endl;
					cout << "Detected Vehicle height: " << vehicle_height << endl;
					cout << "Detected Vehicle x: " << pt1.x << endl;
					cout << "Detected Vehicle y: " << pt1.y << endl;
					cout << "Detected Vehicle contour area " << contourArea(contours[i]) << endl;
					cout << "X difference is " << (abs(vehicles_detected[i][0] - pt1.x) ) << endl;
					cout << "+++++++++++++++++++++++++++++++++++++" << endl;*/
					/*Display the coordinates and features of the newly detected vehicle*/

					/*Find the next empty slot in vehicles memory*/
					int empty_slot = 0;
					for (int i = 0; i < MAX_VEHICLES_IN_FRAME; i++){
						if (vehicles_detected[i][3] == 0 && vehicles_detected[i][2] == 0 && vehicles_detected[i][1]==0){
							empty_slot = i;
							break;
						}
					}

					cout << "~~~~~~~~~~Vehicles currently in memory are : " << endl;
					for (int i = 0; i < vehicles; i++){
						cout << "----------------------- " << i + 1 << "-------------------"<<endl;
						cout << "(x1,y1) -> " << vehicles_detected[i][0] << "," << vehicles_detected[i][1] << endl;
						cout << "(x2,y2) -> " << vehicles_detected[i][4] << "," << vehicles_detected[i][5] << endl;
						cout << "Width = " << vehicles_detected[i][2] << " Height = " << vehicles_detected[i][3] << endl;
					}
					//put or append the vehicle parameters in memory
					vehicles_detected[empty_slot][0] = pt1.x;
					vehicles_detected[empty_slot][1] = pt1.y;
					vehicles_detected[empty_slot][2] = vehicle_width;
					vehicles_detected[empty_slot][3] = vehicle_height;
					vehicles_detected[empty_slot][4] = pt2.x;
					vehicles_detected[empty_slot][5] = pt2.y;

					cout << "~~~~~~~~~~Vehicles currently in memory uodated to : " << ++vehicles << endl;
					pause();
				}
				/*********************check for a new vehicle********************/
				no_of_vehicles_in_frame++;	
			}
			
		}//end of for involving contour check
		//if (frame_no % 2000==0)
		//	pause();
		bool stuck = false;
		/***************check for stuck contour******************/
		if (stuck_frame_count != CONTOUR_STUCK_FRAME_COUNT){
			area[stuck_frame_count] = max_area;
			contour_point[stuck_frame_count][0] = max_area_point.x;
			contour_point[stuck_frame_count][1] = max_area_point.y;
			//cout << "Max contour area detected is " << max_area << " at " << contour_point[stuck_frame_count][0] << endl;
			stuck_frame_count++;
		}
		else if (stuck_frame_count == CONTOUR_STUCK_FRAME_COUNT){
			/*cout << "Re initializing stuck_frame_count" << endl;
			cout << "abs(area[0] - area[1]) is " << abs(area[0] - area[1]) << endl;
			cout << "area[0] is " << area[0] << endl;
			cout << "area[1] is " << area[1] << endl;
			cout << contour_point[0][0] << "," << contour_point[1][0] << "," << contour_point[2][0] << " and area " << area[0] << ", " << area[1]<<endl;
			if ((abs(area[0] - area[1]) >= 0))
				cout << "1 staisfied" << endl;
			if ((abs(area[0] - area[1]) <100))
				cout << "2 staisfied" << endl;*/
			if ((contour_point[0][0] == contour_point[1][0] ) && 
				area[0] != 0 && area[1] != 0 && (abs(area[0] - area[1])>=0) && (abs(area[0] - area[1])<100) ){
				stuck = true;
				cout << "STUCK" << endl;
				background=backup_background.clone();
				//pause();
			}
			stuck_frame_count = 0;
		}
		/***************check for stuck contour******************/

		/********If there is no vehicle found update the no contour frame counter else show the result*********/
		if (!has_potential_vehicle){
			no_contour_frame_vec[no_contour_frame] = nextframe.clone();
			//cout << "No contour frame no: " << no_contour_frame << endl;
			no_contour_frame++;
			imshow("Detected objects", empty);
			cvWaitKey(30);
		}
		else{
			no_contour_frame = 0;
			drawContours(noise_less_frame, contours, MIN_DETECTABLE_AREA, "Detected objects");
			ostringstream v;
			v << no_of_vehicles_in_frame;
			string vehicles = v.str();
			putText(noise_less_frame, vehicles.c_str(), cv::Point(35, 35),
				FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255));
			imshow("Detected objects", noise_less_frame);
			cvWaitKey(30);
		}
		/********If there is no vehicle found update the no contour frame counter else show the result*********/

		/***********Background update starts here************/
		Mat new_background = background.clone();
		Scalar value;
		int pixels_updated = 0;
		if ((!copy_frame_as_background && !stuck) || vehicle_width > frame_width/2){
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
		}
		if (no_contour_frame == 2){
			for (int i = 0; i < MAX_VEHICLES_IN_FRAME; i++){
				for (int j = 0; j < NO_OF_CUES; j++){
					vehicles_detected[i][j] = 0;
				}
			}
			vehicles = 0;
		}
		/*Background cleaning strategy 1: If no contours are reported for a 
		  continuous NO_CONTOUR_FRAME_COUNT then set the current frame as the background frame*/
		if (no_contour_frame == NO_CONTOUR_FRAME_COUNT){
			THRES = getBackgroundThreshold(no_contour_frame_vec);
			outFile << THRES; outFile << endl;
			
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
		/*Background cleaning strategy 1: If no contours are reported for a
		continuous NO_CONTOUR_FRAME_COUNT then set the current frame as the background frame*/

		/**************************************************************************/
		////this algorithm is very expensive
		////fastNlMeansDenoisingColored(new_background, background, 3, 3, 7, 21);
		/**************************************************************************/


		/***************Pause after PAUSE_AFTER_FRAMES frames****************/
		if (frame_no % PAUSE_AFTER_FRAMES == 0){
			cout << "Processed frame no: " << frame_no << endl;
			cout << "No of pixels updated: " << pixels_updated << endl;
			cout << "Total vehicles detected in the video till now:   " << total_vechicles << endl;
			//pause();
		}
		/***************Pause after PAUSE_AFTER_FRAMES frames****************/

		/***************Check for user input******************/
		int x = cvWaitKey(30);
		if (x == 32)
			pause();
		/***************Check for user input******************/
	}//end of while
	cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
	cout << "Total vehicles detected in the video      " << total_vechicles << endl;
	cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
	outFile.close();
	pause();
	return 0;
}//end of main