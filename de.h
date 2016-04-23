#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "memory.h"


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

using namespace std;
using namespace cv;
vector<Vec3f> DE(Mat, Mat, vector<Vec3f>,char[], char[],int,int,int);