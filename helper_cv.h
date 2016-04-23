#include "opencv2/highgui/highgui.hpp";
#include "opencv2/imgproc/imgproc.hpp";
#include "opencv2/photo/photo.hpp";
#include "opencv/cv.h";
#include "opencv/highgui.h";
#include <iostream>;
#include <stdio.h>;
#include <fstream>;
#include <sstream>;
#include <algorithm>;
#include <numeric>;

using namespace std;
using namespace cv;


//Pixel wise comparison model
//Intensity based threshold
const int MIN_INTENSITY_DIFFERENCE = 20;
//Color based threshold
const int MIN_RGB_DIFFERENCE = 8;
const int MIN_VEHICLE_IDENTIFIABLE_DIFFERENCE = 120;
const int PAUSE_AFTER_FRAMES = 100;
const int NO_CONTOUR_FRAME_COUNT = 3;
const int CONTOUR_STUCK_FRAME_COUNT = 3;
const int TRAINING_FRAMES = 3;
const int MAX_VEHICLES_IN_FRAME = 5;
const int NO_OF_CUES = 6;
const int MIN_WIDTH_TO_HEIGHT_RATIO = 2;
const int MAX_WIDTH_TO_HEIGHT_RATIO = 5;

//Histogram based model
const int INITIAL_LEARN_FRAME_LIMIT = 0;
const int LEARN_FRAME_LIMIT_LESS_THRES = 4;
const int LEARN_FRAME_LIMIT_GREATER_THRES = 2;
const int CHISQUARE = 1;
const int BHATTA = 3;
const int INTERSEC = 2;

//Vehicle identification parameters
const int MIN_VEHICLE_HEIGHT = 8;
const int MIN_VEHICLE_WIDTH = 20;
const double MIN_VEHICLE_AREA = 150;
const double MIN_DETECTABLE_AREA = 50;
const int MAX_Y_TOLERANCE = 10;
const int MAX_X_TOLERANCE = 10;
const int ENTRY_EXIT_TOLERANCE = MIN_VEHICLE_WIDTH;
const int MAX_HEIGHT_TOLERANCE = 5;
const int MAX_WIDTH_TOLERANCE = 5;

//Circle detection parameters
const int cannyThresholdInitialValue = 1;
const int accumulatorThresholdInitialValue = 12;
const int maxAccumulatorThreshold = 400;
const int maxCannyThreshold = 255;
const int inverseResolutionInitialValue = 2;
const int maxInverseResolution = 3;
const int minRadius = 3;
const int maxRadius = 10;


const int MAX_CONTOURS = 10;
const int APERTURE = 3;
const int MIN_CONTOUR_AREA = 50;

//Window display parameters
const int ORIGIN = 0;
const int SHIFT = 520;
const int WINDISTANCE = 20;
const int max_rows = 300;
const int max_cols = 400;

//crop parameters
const bool ROI_ENABLED = FALSE;
const double cropY = 0.10;
const double cropHeight = 0.50;

//color weights
const double R_WEIGHT = 0.33;
const double G_WEIGHT = 0.33;
const double B_WEIGHT = 0.34;



void pause();
int getBackgroundThreshold(VideoCapture cap, int limit);
int getBackgroundThreshold(vector<Mat>);
int getThrs(vector<int>);
void createNMoveWindow(char name[], int width, int height, Mat data, int x, int y);
vector<Mat> getNormalizedHist(Mat frame, int histSize, float range[], bool uniform, bool accumulate);
double compareHistograms(Mat a, Mat b, int measure);
void drawContours(Mat src, vector<vector<Point>> contours, double max_area, char*);
void drawContours(Mat src, vector<vector<Point>> contours);
void drawContours(Mat src, vector<vector<Point>> contours);
void drawContoursNDeNoise(Mat src, vector<vector<Point>> contours, double min_area);
void drawContoursNDeNoise1(Mat src, vector<vector<Point>> contours, double min_area);
Mat drawContoursNDeNoise2(Mat src, vector<vector<Point>> contours, double min_area);
Mat drawContoursNDeNoise2Coloured(Mat src, Mat pixe_src, vector<vector<Point>> contours, double min_area);
void findOnes(Mat src);
double mean(vector<double> arr);
double stdev(double mean, vector<double> arr);
Mat emptyMat(Mat i);
bool checkCircles(Mat src_gray);
bool HoughDetection(const Mat& src_gray, const Mat& src_display, int cannyThreshold, int accumulatorThreshold, int inverseResolution, int, int);