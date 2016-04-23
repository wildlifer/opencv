#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
char key;
int main()
{
	cvNamedWindow("Camera_Output", 1);    //Create window
	CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);  //Capture using any camera connected to your system
	while (1){ //Create infinte loop for live streaming

		IplImage* frame = cvQueryFrame(capture); //Create image frames from capture
		cvShowImage("Camera_Output", frame);   //Show image frames on created window
		key = cvWaitKey(100);     //Capture Keyboard stroke
		if (char(key) == 27){
			break;      //If you hit ESC key loop will break.
		}
	}
	cvReleaseCapture(&capture); //Release capture.
	cvDestroyWindow("Camera_Output"); //Destroy Window
	return 0;
}