#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

int g_sliderPos = 0;
CvCapture* g_capture = NULL;

void onTrackbarSlide(int pos) {
	cvSetCaptureProperty(
		g_capture,
		CV_CAP_PROP_POS_FRAMES,
		pos
		);
}
int main(int argc, char** argv)
{
	cvNamedWindow("Example3", CV_WINDOW_AUTOSIZE);
	IplImage* frame;// = cvLoadImage("images/highway/bus1.png");
	g_capture = cvCreateFileCapture("videos/6.avi");
	int frames = (int)cvGetCaptureProperty(
						g_capture,
						CV_CAP_PROP_FRAME_COUNT
						);
	if (frames != 0) {
		cvCreateTrackbar(
			"Position",
			"Example3",
			&g_sliderPos,
			frames,
			onTrackbarSlide
			);
	}
	std::cout << "No of frames are " << frames;
	while (1){
		frame = cvQueryFrame(g_capture);
		if (!frame) break;
		cvShowImage("Example3",frame);
		char c = cvWaitKey(100);
		if (c == 27) break;
	}

	//cvWaitKey(0);
}