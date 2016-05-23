
#include "de.h"
ofstream out;
String base = "images/CEC2015/Base/";
String writebase = "images/CEC2015/";
using namespace std;
using namespace cv;



ofstream outFile;
ofstream outFile2;
ofstream outFile3;
ofstream outFile4;

Mat src; Mat src_gray; Mat src_gray1;
int thresh = 120;
int max_thresh = 255;
RNG rng(12345);

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
	const int cannyThresholdInitialValue = 100;
	const int accumulatorThresholdInitialValue = 20;
	const int maxAccumulatorThreshold = 400;
	const int maxCannyThreshold = 255;
	const int inverseResolutionInitialValue = 1;
	const int maxInverseResolution = 1;
	int THRESHOLD = 22;
	//Mat to display the type of vehicle

	vector<Vec3f> HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold, int inverseResolution)
	{
		// will hold the results of the detection
		std::vector<Vec3f> circles;
		// runs the actual detection
		//HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 8, cannyThreshold, accumulatorThreshold, 14, 40);
		HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, inverseResolution, src_gray.rows / 14, cannyThreshold, accumulatorThreshold, 14, 35);
		return circles;
	}
	void classify(){
	}
}
void ShowResults(vector<Vec3f> circles, Mat src_display, String file){
	Mat vehicle = Mat::zeros(200, 1260, CV_8UC3);
	Mat clearedVehicle = Mat::zeros(200, 1260, CV_8UC3);
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
	case 0:outFile << " False - " << circles.size() << endl;
		putText(clearedVehicle, "------", Point(vehicle.cols / 3 - vehicle.cols / 10, vehicle.rows / 2),
			FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
		break;
	case 1:outFile << " False - " << circles.size() << endl;
		putText(vehicle, "False", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
			FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
		break;
	case 2:outFile << " Car - " << circles.size() << endl;
		putText(vehicle, "Car", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
			FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
		break;
	case 3:outFile << " Car - " << circles.size() << endl;
		putText(vehicle, "Car", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
			FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false); break;
	case 4:outFile << " Truck - " << circles.size() << endl;
		putText(vehicle, "Truck", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
			FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
		break;
	case 5:outFile << " Truck - " << circles.size() << endl;
		putText(vehicle, "Truck 1", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
			FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
		break;
	case 6:outFile << " Truck - " << circles.size() << endl;
		putText(vehicle, "Truck 2", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
			FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
		break;
	default: outFile << " SpaceShip - " << circles.size() << endl;
		putText(vehicle, "SpaceShip", Point(vehicle.cols / 2 - vehicle.cols / 10, vehicle.rows / 2),
			FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
		break;
	}

	// shows the results
	imshow("Manual Detection", display);
	cvMoveWindow("Manual Detection", 0, 0);
	//cvWaitKey(0);
	//cv::moveWindow("Original Video", src_display.cols, 0);
	//imshow("Detected Vehicle", vehicle);
	//cvMoveWindow("Detected Vehicle", 0, 450);
	//cv::moveWindow("Detected Vehicle", 0, src_display.rows + 30);
	//cvWaitKey(1000);
	//putText(clearedVehicle, "------", Point(vehicle.cols / 3 - vehicle.cols / 10, vehicle.rows / 2),
		//FONT_HERSHEY_SIMPLEX, 4, Scalar(255, 255, 255), 6, 8, false);
	//imshow("Detected Vehicle", clearedVehicle);
	//cvMoveWindow("Detected Vehicle", 0, 450);
}

vector<Vec3f> GetAlignedCenters(vector<Vec3f> circles){
	const int noOfCircles = circles.size();
	int noOfFinalCircles = 0;
	int **adjacency = new int*[noOfCircles];
	for (int i = 0; i < noOfCircles; i++)
		adjacency[i] = new int[noOfCircles];

	//Discard the falsely detected circles
	int* adjacencySum = new int[noOfCircles];
	for (int i = 0; i < noOfCircles; i++){
		adjacencySum[i] = 0;
	}
	for (int i = 0; i < noOfCircles; i++){
		//cout << "Circle No " << i << " with center "<<
		for (int j = 0; j < noOfCircles; j++){
			if (i == j){
				adjacency[i][j] = 1;
			}
			else if (abs(cvRound(circles[i][1]) - cvRound(circles[j][1])) < THRESHOLD) {
				adjacency[i][j] = 1;
			}
			else{
				adjacency[i][j] = 0;
			}
			cout << adjacency[i][j] << ",";
		}
		cout << endl;
	}
	//sum the adjacency matrix
	for (int i = 0; i< noOfCircles; i++){
		for (int j = 0; j < noOfCircles; j++){
			adjacencySum[i] += adjacency[i][j];
		}
	}
	//find the max
	int max = adjacencySum[0];
	for (int i = 1; i < noOfCircles; i++){
		if (adjacencySum[i]>max)
			max = adjacencySum[i];
	}
	int j = 0;
	for (int i = 0; i < noOfCircles; i++){
		if (adjacencySum[i] == max){
			j++;
		}
	}
	vector<Vec3f> newCenters(j);
	j = 0;


	for (int i = 0; i < noOfCircles; i++){
		if (adjacencySum[i] == max){
			cout << " i is " << i << " and j is " << j << endl;
			newCenters.at(j) = circles.at(i);
			j++;
		}
	}

	return newCenters;
}
void ShowCircles(char str[], vector<Vec3f> circles, Mat src_display,String dir, String file){
	Mat vehicle = Mat::zeros(200, 1260, CV_8UC3);
	Mat clearedVehicle = Mat::zeros(200, 1260, CV_8UC3);
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

	// shows the results
	cvNamedWindow(str, 0);
	cvResizeWindow(str, 600, 400);
	imshow(str, display);

	//write
	String path = writebase + dir + "/" + file;
	imwrite(path, display);
	cout << "Writing ->> " << path << endl;
	//cvWaitKey(0);
	cvMoveWindow(str, 600, 0);
	cvWaitKey(1000);
}
void OptimizeDE(Mat src, Mat src_gray, vector<Vec3f> circles, char infile[], char outfile[], String file, int imageNo){
	//vector<String> strategies = { "DEBest1Bin", "DERand1Bin", "DERandToBest1Bin", "DEBest2Bin", "DERand2Bin" };
	vector<String> strategies = { "DEBest1Bin" };
	String outFile = writebase + "/out.txt";
	char *out = (char*)outFile.c_str();
	//memcpy(outfile,outFile.c_str(),outFile.size());
	for (int i = 0; i < strategies.size(); i++){
		vector<Vec3f> deCircles = DE(src, src_gray, circles, infile, out, i + 6, strategies.size(), imageNo);
		ShowCircles("Detected by DE", deCircles, src, strategies[i],file);
	}
	//cvWaitKey(0);
}
void DetectCircles(Mat src,String file,int imageNo){
	Mat src_gray;
	cvtColor(src, src_gray, CV_BGR2GRAY);
	// Reduce the noise so we avoid false circle detection
	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
	//declare and initialize both parameters that are subject to change
	int cannyThreshold = cannyThresholdInitialValue;
	int accumulatorThreshold = accumulatorThresholdInitialValue;
	int inverseResolution = inverseResolutionInitialValue;
	cannyThreshold = std::max(cannyThreshold, 1);
	accumulatorThreshold = std::max(accumulatorThreshold, 1);
	std::vector<Vec3f> circles = HoughDetection(src_gray, src, cannyThreshold, accumulatorThreshold, inverseResolution);
	std::vector<Vec3f> alignedCircles = GetAlignedCenters(circles);
	ShowResults(circles, src,file);
	//Call DE module
	OptimizeDE(src, src_gray, alignedCircles, "in.txt", "out.txt", file, imageNo);
	//cvWaitKey(10);
}

int main(int argc, char** argv)
{
	cvNamedWindow("Manual Detection", 0);
	cvResizeWindow("Manual Detection", 600, 400);
	String image = "";
	ifstream files("images/CEC2015/Base/list.txt");
	//clear content of output file
	FILE  *fpout_ptr;
	fpout_ptr = fopen("images/CEC2015/out.txt", "w");
	fclose(fpout_ptr);
	int imageNo = 0;
	if (files.is_open())
	{
		while (getline(files, image))
		{
			cout << image << '\n';
			String path = base + image;
			Mat background, hsv_background;
			background=imread(path, CV_LOAD_IMAGE_COLOR);
			imshow("Manual Detection", background);
			cvMoveWindow("Manual Detection", 0, 0);
			outFile.open("data/2.1.txt");

			//Mat roi(background, Rect(0, 2 * background.rows / 5,background.cols, background.rows / 3));
			//DetectCircles(roi);
			//cvNamedWindow("ROI", 0);
			//cvResizeWindow("Original Video", 600, 400);
			//cvMoveWindow("ROI", 0, 0);
			//imshow("ROI", roi);
			DetectCircles(background, image, imageNo);
			imageNo++;
		}
		files.close();
	}
	outFile.close();
	int a = cvWaitKey(0);
	return 0;
}