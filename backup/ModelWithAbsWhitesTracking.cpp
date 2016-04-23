#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
IplImage* background;
ofstream outFile;

long threshouldWhiteDifference = 3000;
long threshouldGrayWhiteDifference = 1;
long noisyWhitesThreshould = 2000;
long grayNoisyWhitesThreshould = 5;
void AllocateImages(IplImage* I){


}
int ObjectDetected(){

	return 1;
}
bool Checkbuffer(long initW, long currentW){
	if ((initW - currentW) < noisyWhitesThreshould)
		return false;
}
namespace
{
	// windows and trackbars name
	const std::string windowName = "Hough Circle Detection";
	const std::string detectedVehicle = "Detected Vehicle";
	const std::string cannyThresholdTrackbarName = "Canny threshold";
	const std::string accumulatorThresholdTrackbarName = "Accumulator Threshold";
	const std::string inverseResolutionName = "Resolution";
	const std::string usage = "Usage : tutorial_HoughCircle_Demo <path_to_input_image>\n";

	// initial and max values of the parameters of interests.
	const int cannyThresholdInitialValue = 65;
	const int accumulatorThresholdInitialValue = 20;
	const int maxAccumulatorThreshold = 400;
	const int maxCannyThreshold = 255;
	const int inverseResolutionInitialValue = 1;
	const int maxInverseResolution = 4;
	//Mat to display the type of vehicle


	void HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold, int inverseResolution)
	{
		Mat vehicle = Mat::zeros(200, 1260, CV_8UC3);
		Mat clearedVehicle = Mat::zeros(200, 1260, CV_8UC3);
		// will hold the results of the detection
		std::vector<Vec3f> circles;
		// runs the actual detection
		HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 8, cannyThreshold, accumulatorThreshold, 14, 30);

		/// Apply the Hough Transform to find the circles
		//HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1.98, src_gray.rows / 5, 250, 50, 0, 30);

		// clone the colour, input image for displaying purposes
		Mat display = src_display.clone();
		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle center
			circle(display, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// circle outline
			circle(display, center, radius, Scalar(0, 0, 255), 3, 8, 0);
		}
		switch (circles.size()){
		case 1:outFile << " False - " << circles.size() << endl;
			putText(vehicle, "False", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
			break;
		case 2:outFile << " Car - " << circles.size() << endl;
			putText(vehicle, "Car", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
			break;
		case 3:outFile << " Car - " << circles.size() << endl;
			putText(vehicle, "Truck", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false); break;
		case 4:outFile << " Truck - " << circles.size() << endl;
			putText(vehicle, "Truck", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
			break;
		default: outFile << " SpaceShip - " << circles.size() << endl;
			putText(vehicle, "SpaceShip", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
				FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
			break;
		}

		// shows the results
		imshow("Original Video", display);
		cvMoveWindow("Original Video", 0, 0);
		//cv::moveWindow("Original Video", src_display.cols, 0);
		imshow("Detected Vehicle", vehicle);
		cvMoveWindow("Detected Vehicle", 0, 450);
		//cv::moveWindow("Detected Vehicle", 0, src_display.rows + 30);
		cvWaitKey(1000);
		putText(clearedVehicle, "------", Point(vehicle.cols / 3 - vehicle.cols / 10, vehicle.rows / 2),
			FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
		imshow("Detected Vehicle", clearedVehicle);
		cvMoveWindow("Detected Vehicle", 0, 450);
	}
}
int main(int argc, char** argv)
{
	CvCapture* capture = cvCreateFileCapture("videos/1.4.mp4");
	long initWhites = 0;
	long whites = 0;
	long initWhitesGray = 0;
	long whitesGray = 0;
	long diffWhitesGray = 0;

	long whitesPrevFrame = 0;
	long whiteCurrentDiffFrame = 0;
	long whitePrevDiffFrame = 0;

	long oldBackWhites = 0;

	std::map<long, IplImage*> m;

	IplImage* newFrame = cvQueryFrame(capture);
	background = cvCreateImage(cvSize(newFrame->width, newFrame->height), IPL_DEPTH_8U, 3);
	IplImage*  frame = cvCreateImage(cvSize(newFrame->width, newFrame->height), IPL_DEPTH_8U, 3);
	IplImage* diffFrame;
	IplImage* diffFirstFrame;

	IplImage* oldFrameGray;
	IplImage* frameGray;
	IplImage* diffFrameGray;

	IplImage* oldFrameBinary;
	IplImage* frameBinary;
	IplImage* diffFrameBinary;

	IplImage* backgroundBinary;

	CvSize sz = cvGetSize(frame);
	//RBG frames
	diffFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	diffFirstFrame = cvCreateImage(sz, IPL_DEPTH_8U, 3);
	//gray frames
	oldFrameGray = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	frameGray = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	diffFrameGray = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	//binary frames
	frameBinary = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	diffFrameBinary = cvCreateImage(sz, IPL_DEPTH_8U, 1);
	backgroundBinary = cvCreateImage(sz, IPL_DEPTH_8U, 1);

	cvResize(newFrame, frame, 3);

	int b = 0;
	bool first = true;
	bool second = true;

	cvNamedWindow("Original Video", 0);
	cvResizeWindow("Original Video", 600, 400);

	cvNamedWindow("Diff Frame", 0);
	cvResizeWindow("Diff Frame", 600, 400);
	bool buffer = true;
	int pause = 0;
	while (1) {
		//pause = cvWaitKey(100);
		//if (pause == 32)
		//cvWaitKey();
		newFrame = cvQueryFrame(capture);
		if (!newFrame) break;
		cvCvtColor(newFrame, frameGray, CV_BGR2GRAY);
		cvThreshold(frameGray, frameBinary, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		if (first){
			diffFrame = cvCloneImage(frame);
			diffFirstFrame = cvCloneImage(frameBinary);

			oldFrameGray = cvCloneImage(frameGray);
			oldFrameBinary = cvCloneImage(frameBinary);

			first = false;
			initWhites = cvCountNonZero(frameBinary);
			initWhitesGray = cvCountNonZero(frameGray);
			//cvConvertScale(frameGray, oldFrameGray, 1.0, 0.0);
			cout << "Init whites are " << (initWhites) << endl;
			//cvWaitKey(0);
			continue;
		}


		whites = cvCountNonZero(frameBinary);
		whitesGray = cvCountNonZero(frameGray);
		cvAbsDiff(oldFrameBinary, frameBinary, diffFrameBinary);
		//cvAbsDiff(oldFrameGray, frameGray, diffFrameGray);
		whiteCurrentDiffFrame = cvCountNonZero(diffFrameBinary);
		//diffWhitesGray = cvCountNonZero(diffFrameGray);
		//cout << "Current Frame whites are " << (whites) << endl;
		cout << "==================================================================" << endl;
		cout << "Current  binary whites are " << (whites) << endl;
		cout << "Change in absolute binary whites are " << (initWhites - whites) << endl;
		//cout << "Diff whites binary are " << cvCountNonZero(diffFrameBinary) << endl;
		cout << "Change in relative whites are " << abs(whitesPrevFrame - whites) << endl;
		cout << "(initWhitesGray - whitesGray) is ->> " << (initWhitesGray - whitesGray) << endl;
		cout << "Buffer is " << buffer << endl;
		//cout << "Diff GRAY whites are " << diffWhitesGray << endl;
		//cout << endl;
		//cout << "Diff frame whites are " << cvCountNonZero(diffFrame) << endl;
		//cvWaitKey(0);
		if (abs(initWhites - whites) < noisyWhitesThreshould){// || (initWhitesGray - whitesGray) < grayNoisyWhitesThreshould){
			buffer = false;
			//oldBackWhites = 0;
			cout << "********Toggling buffer to false with abs(initWhites - whites)->> " << abs(initWhites - whites) << endl;
			cout << "******* and (initWhitesGray - whitesGray)->> " << (initWhitesGray - whitesGray) << endl;
			//continue;
		}
		//buffer = Checkbuffer(initWhites, whites);
		//if ((initWhitesGray - whitesGray) < grayNoisyWhitesThreshould){
			//oldBackWhites = 0;
		//}
		//if ((initWhites - whites) > threshouldWhiteDifference && (initWhitesGray - whitesGray) >  threshouldGrayWhiteDifference && buffer == false){//Check 1
		if (abs(initWhites - whites) > threshouldWhiteDifference  && abs(whitesPrevFrame - whites)>300 && buffer == false){//Check 1
			cout << "+++++++++++++++++++++" << endl;
			cout << "change detected" << endl;
			cout << "+++++++++++++++++++++" << endl;
			//cvWaitKey(0);
			//cout << "Current Frame whites are " << (whites) << endl;
			//cout << "Current Diff Frame whites are " << whiteCurrentDiffFrame << endl;
			cout << "abs(whitesPrevFrame - whites) are  " << abs(whitesPrevFrame - whites) << endl;
			cout << "abs(oldBackWhites - abs(whitesPrevFrame - whites) are  " << abs(oldBackWhites - abs(whitesPrevFrame - whites)) << endl;
			//cout << "Previous Frame whites are " << (whitesPrevFrame) <<  endl;
			//if (abs(whites - whitesPrevFrame) <= 50 && abs(whitePrevDiffFrame > whiteCurrentDiffFrame) > 1000){
			cout << "PREVIOUS - INIT WHITES ->> " << abs(whitesPrevFrame - initWhites) << endl;
			cout << "CURRENT - INIT WHITES ->> " << abs(whites - initWhites) << endl;
			if (abs(whitesPrevFrame - initWhites) > abs(whites - initWhites)){
				cout << "*****First Condition satisfied*********" << endl;
				//cout << "PREVIOUS - INIT WHITES ->> " << abs(whitesPrevFrame - initWhites) << endl;
				//cout << "CURRENT - INIT WHITES ->> " << abs(whites - initWhites) << endl;
				//cvWaitKey(0);
				if (abs(whitesPrevFrame - whites) < 700){
					cout << "*****Second Condition satisfied*********" << endl;
					//cvWaitKey(0);
						if (abs(oldBackWhites - abs(whitesPrevFrame - whites))>165){
						cout << "***********************" << endl;
						cout << "Best frame detected" << endl;
						cout << "***********************" << endl;
						cout << "(whitesPrevFrame - whites) is " << (whitesPrevFrame - whites) << endl;
						backgroundBinary = cvCloneImage(frameBinary);
						oldBackWhites = abs(whitesPrevFrame - whites);
						//oldBackWhites = abs(whites);
						Mat src(newFrame), src_gray(frameGray);
						// Reduce the noise so we avoid false circle detection
						GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
						//declare and initialize both parameters that are subjects to change
						int cannyThreshold = cannyThresholdInitialValue;
						int accumulatorThreshold = accumulatorThresholdInitialValue;
						int inverseResolution = inverseResolutionInitialValue;

						// create the main window, and attach the trackbars
						//namedWindow(windowName, WINDOW_AUTOSIZE);
						//cvNamedWindow("Original Video", 0);
						//cvResizeWindow("Original Video", 600, 600);
						cannyThreshold = std::max(cannyThreshold, 1);
						accumulatorThreshold = std::max(accumulatorThreshold, 1);
						HoughDetection(src_gray, src, cannyThreshold, accumulatorThreshold, inverseResolution);
						buffer = true;
						cvWaitKey(0);
					}
				}
			}
		}
		whitesPrevFrame = whites;
		whitePrevDiffFrame = whiteCurrentDiffFrame;
		//cvSmooth(diffFrame, diffFrame, CV_BLUR);
		//cvThreshold(diffFrame, diffFrame, 25, 255, CV_THRESH_BINARY);
		cvShowImage("Original Video", newFrame);
		cvMoveWindow("Original Video", 0, 0);
		//Calculate diff

		cvShowImage("Diff Frame", diffFrameBinary);
		cvMoveWindow("Diff Frame", (newFrame->width / 3) + 10, 0);
		//New Background
		cvConvertScale(frameGray, oldFrameGray, 1.0, 0.0);

		pause = cvWaitKey(20);
		//cout << "Pause is" << pause << endl;
		if (pause == 32){
			pause = cvWaitKey(0);
		}
		//cout << "Pause after is " << pause << endl;
	}
	int a = cvWaitKey(0);
	cvReleaseCapture(&capture);
	return 0;
}