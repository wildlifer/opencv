#include "opencv2/highgui/highgui.hpp"
int main(int argc, char** argv) {
	cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);
	CvCapture* capture = cvCreateFileCapture("videos/7.avi");
	IplImage* frame;
	while (1) {
		frame = cvQueryFrame(capture);

		if (!frame) break;
		cvShowImage("Example2", frame);
		char c = cvWaitKey(300);
		if (c == 27) break;
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("Example2");
}