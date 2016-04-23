#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;

IplImage* getBestFrame(map<long,IplImage*> m){
	IplImage* frame = NULL;
	map<long,IplImage*>::iterator it;
	long mean = 0L;
	long key = 0L;
	int i = 0;
	for (it = m.begin(), i = 0; it != m.end(); it++,i++){
		mean += it->first;
		if (i == m.size() / 2+1){
			key = it->first;
			break;
		}
	}
	mean = (long) mean / m.size();
	cout << "Mean is " << mean << endl;
	cout << "Best found at " << key << endl;
	cvShowImage("Best1", m[key]);
	return frame=m[key];
}
int main(int argc, char** argv)
{
	cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);
	//cvNamedWindow("Best", CV_WINDOW_AUTOSIZE);
	std::map<long,IplImage*> m;
	ofstream outFile;
	int a = 0;
	outFile.open("data\\frameWhites.txt");
	//cvNamedWindow("Example3", CV_WINDOW_AUTOSIZE);
	CvCapture* capture = cvCreateFileCapture("videos/6.avi");
	long whites = 0;
	IplImage* frame = cvQueryFrame(capture);
	IplImage* frameGray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	IplImage* bestFrame = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
	cvCvtColor(frame, frameGray, CV_BGR2GRAY);
	long initialWhites = cvCountNonZero(frameGray);
	long thresholdWhitecount = 70;
	int counter = 0;
	while ((frame = cvQueryFrame(capture))!=NULL) {
		//if (!frame) break;
		cvCvtColor(frame, frameGray, CV_BGR2GRAY);
		
		whites = cvCountNonZero(frameGray);
		//cout << 
		if ((initialWhites - whites) < thresholdWhitecount) continue;
		if (++counter == 20){
			counter = 0;
			bestFrame = getBestFrame(m);
			
		}		
		m[whites] = frameGray;
		cvShowImage("Example2", frameGray);
		cvShowImage("Map", m[whites]);
		//cvShowImage("Best", bestFrame);
		cout << " No of whites are " << whites << endl;
		outFile << whites << endl;
		//cin >> a;
		char c = cvWaitKey(333);
		if (c == 27) break;
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("Example2");
	cvWaitKey(0);
	return 0;
}