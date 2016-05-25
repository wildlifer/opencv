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
#include "stdlib.h"
#include "math.h"
#include "memory.h"


using namespace std;
using namespace cv;

//Video
const String video = "videos/4.5.mp4";
//Pixel wise comparison model
//Intensity based threshold
const int MIN_INTENSITY_DIFFERENCE = 20;
//Color based threshold
const int MIN_RGB_DIFFERENCE = 8;
const int MIN_VEHICLE_IDENTIFIABLE_DIFFERENCE = 120;
const int PAUSE_AFTER_FRAMES = 100;
const int NO_CONTOUR_FRAME_COUNT = 3;
const int CONTOUR_STUCK_FRAME_COUNT = 3;
const int TRAINING_FRAMES = 5;
const int MAX_VEHICLES_IN_FRAME = 5;
const int NO_OF_CUES = 7;
/*
Cue 1 : left x coordinate
Cue 2: left y coordinate
Cue 3: width
Cue 4: height
Cue 5: right x coordinate
Cue 6: right y coordinate
Cue 7: direction (1 for left, 2 for right)
*/
const int MERGE_THRES = 5;
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
const int MAX_Y_TOLERANCE = 15;
const int MAX_X_TOLERANCE = 10;
const int ENTRY_EXIT_TOLERANCE = MIN_VEHICLE_WIDTH;
const int MAX_HEIGHT_TOLERANCE = 5;
const int MAX_WIDTH_TOLERANCE = 5;

//Circle detection parameters
const int cannyThresholdInitialValue = 50;
const int accumulatorThresholdInitialValue = 10;
const int inverseResolutionInitialValue = 1;
const int minRadius = 5;
const int maxRadius = 10;
const int HOUGH_THRESHOLD = 5; //dont know its use yet

const int maxAccumulatorThreshold = 400;
const int maxCannyThreshold = 255;
const int maxInverseResolution = 3;

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

//DE related
#define MAXPOP  500
#define MAXDIM  35

/*------Constants for rnd_uni()--------------------------------------------*/

#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#define ONE 1
#define ZERO 0

//opencv constants
#define MAXCIRCLES 10
#define CENTERTHRES 2
#define YTHRESHOLD 4
#define RADIUSTHRES 6
#define MINCENTERDISTANCE 40
#define GLBMIN 2
#define EPSILON 0.0000000001

//Functions
void pause();
int getBackgroundThreshold(VideoCapture cap, int limit);
double getBackgroundThreshold(vector<Mat>);
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
vector<Vec3f> GetAlignedCenters(vector<Vec3f> circles);
void drawCircles(Mat, std::vector<Vec3f> , char*,  int, int);
void OptimizeDE(Mat src, Mat src_gray, vector<Vec3f> circles, char infile[], char outfile[], String file, int imageNo);
vector<Vec3f> DE(Mat, Mat, vector<Vec3f>, char[], char[], int, int, int);