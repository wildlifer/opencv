#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>
int main(int argc, char** argv){
	IplImage* big = cvLoadImage("images/car2.jpg");
	IplImage* img = cvLoadImage("images/car1.jpg");
	cvShowImage("Original",big);
	float scale = (float)big->width / 300;
	//cvSetImageROI(big, cvRect(big->width/2, big->height/2, (int)(big->width / scale), (int)(big->height / scale)));
	cvSetImageROI(big, cvRect(big->width/2, 0, (int)(big->width /2), (int)(big->height)));
	//cvShowImage("ROI-Set", big);
	cvResize(img, big);
	
	cvResetImageROI(big);
	//cvShowImage("Resized", img);
	cvShowImage("ROI-ReSet", big);
	cvMoveWindow("Original",0,0);
	cvMoveWindow("ROI-ReSet", big->width, 0);
	cvWaitKey();
}
