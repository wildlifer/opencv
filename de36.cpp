
/**H*O*C**************************************************************
**                                                                  **
** No.!Version! Date ! Request !    Modification           ! Author **
** ---+-------+------+---------+---------------------------+------- **
**  1 + 3.1  +5/18/95+   -     + strategy DE/rand-to-best/1+  Storn **
**    +      +       +         + included                  +        **
**  1 + 3.2  +6/06/95+C.Fleiner+ change loops into memcpy  +  Storn **
**  2 + 3.2  +6/06/95+   -     + update comments           +  Storn **
**  1 + 3.3  +6/15/95+ K.Price + strategy DE/best/2 incl.  +  Storn **
**  2 + 3.3  +6/16/95+   -     + comments and beautifying  +  Storn **
**  3 + 3.3  +7/13/95+   -     + upper and lower bound for +  Storn **
**    +      +       +         + initialization            +        **
**  1 + 3.4  +2/12/96+   -     + increased printout prec.  +  Storn **
**  1 + 3.5  +5/28/96+   -     + strategies revisited      +  Storn **
**  2 + 3.5  +5/28/96+   -     + strategy DE/rand/2 incl.  +  Storn **
**  1 + 3.6  +8/06/96+ K.Price + Binomial Crossover added  +  Storn **
**  2 + 3.6  +9/30/96+ K.Price + cost variance output      +  Storn **
**  3 + 3.6  +9/30/96+   -     + alternative to ASSIGND    +  Storn **
**  4 + 3.6  +10/1/96+   -    + variable checking inserted +  Storn **
**  5 + 3.6  +10/1/96+   -     + strategy indic. improved  +  Storn **
**                                                                  **
***H*O*C*E***********************************************************/

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h";
#include "opencv/highgui.h"
# include "de.h"
#include <algorithm>

using namespace std;
using namespace cv;

/*------------------------Macros----------------------------------------*/

/*#define ASSIGND(a,b) memcpy((a),(b),sizeof(double)*D) */  /* quick copy by Claudio */
/* works only for small  */
/* arrays, but is faster.*/

/*------------------------Globals---------------------------------------*/

long  rnd_uni_init;                 /* serves as a seed for rnd_uni()   */
double c[MAXPOP][MAXDIM], d[MAXPOP][MAXDIM];
double(*pold)[MAXPOP][MAXDIM], (*pnew)[MAXPOP][MAXDIM], (*pswap)[MAXPOP][MAXDIM];
int reset = 0;
vector<int> axles = { 2, 2, 2, 2, 2, 2, 2, 4, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2 };
//vector<int> axles = {  5, 5 };


/*---------Function declarations----------------------------------------*/

void  assignd(int D, double a[], double b[]);
double rnd_uni(long *idum);    /* uniform pseudo random number generator */
vector<Vec3f> extern evaluate(Mat src, Mat src_gray,int D, double tmp[], long *nfeval); /* obj. funct. */
double inibound_h[] = { 4, 20, 200, 100 };      /* upper parameter bound              */
//double inibound_l;      /* lower parameter bound              */
double inibound_l[] = { 1, 10, 40, 20 };
/*---------Function definitions-----------------------------------------*/
void pause(char D[])
{
	int s = 0;
	printf("%s", D);
	scanf("%d", &s);
}
void plotCircles(Mat src_display,vector<Vec3f> circles){
	cvNamedWindow("DE detected circles", 0);
	cvResizeWindow("DE detected circles", 600, 400);
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
	imshow("DE detected circles", display);
	cvMoveWindow("DE detected circles", 600, 0);
	cvWaitKey(0);
}
int checkExtraCentroids(Mat src,vector<Vec3f> circles, vector<Vec3f> deCircles){
	int noOfCircles = 0;
	vector<Vec3f> extraCircles(10);
	int extra = 0;
	for (int j = 0; j < deCircles.size();j++){
		cout << endl;
		if (abs(cvRound(deCircles[j][1]) - cvRound(circles[0][1])) <= CENTERTHRES && abs(cvRound(deCircles[j][2]) - cvRound(circles[0][2])) < RADIUSTHRES){
			//cout << " Extra centroid found: Original is  " << circles[0][1] << " radius = " << circles[0][2]<< " and found by DE is " << deCircles[j][1] << " with radius = "<< deCircles[j][2]<<endl;
			extraCircles[extra++] = deCircles[j];
			noOfCircles++;
		}	
	}
	plotCircles(src,extraCircles);
	return noOfCircles-circles.size();
}
bool checkCenteroidMatch(vector<Vec3f> circles, vector<Vec3f> deCircles){
	bool match = false;
	cout << endl;
	for (int i = 0; i < circles.size(); i++){
		match = false;
		for (int j = 0; j < circles.size(); j++){
			if ( cvRound(circles[i][1])==cvRound(deCircles[j][1]) ){
				//cout << "Match Found with " << circles[i][1] << " and " << (deCircles[j][1])<< endl;
				match = true;
				//pause("Matched");
				break;
			}
		}
	}
	if (match == false){
		//cout << "Equal no. of circles but centroids dont match "  << endl;
		//pause("");
	}
	return match;
}
void  assignd(int D, double a[], double b[])
/**C*F****************************************************************
**                                                                  **
** Assigns D-dimensional vector b to vector a.                      **
** You might encounter problems with the macro ASSIGND on some      **
** machines. If yes, better use this function although it's slower. **
**                                                                  **
***C*F*E*************************************************************/
{
	int j;
	for (j = 0; j<D; j++)
	{
		a[j] = b[j];
		//check bounds
		if (a[j] < inibound_l[j])
			a[j] = inibound_l[j];
		if (a[j] > inibound_h[j])
			a[j] = inibound_h[j];
	}
}


double rnd_uni(long *idum)
/**C*F****************************************************************
**                                                                  **
** SRC-FUNCTION   :rnd_uni()                                        **
** LONG_NAME      :random_uniform                                   **
** AUTHOR         :(see below)                                      **
**                                                                  **
** DESCRIPTION    :rnd_uni() generates an equally distributed ran-  **
**                 dom number in the interval [0,1]. For further    **
**                 reference see Press, W.H. et alii, Numerical     **
**                 Recipes in C, Cambridge University Press, 1992.  **
**                                                                  **
** FUNCTIONS      :none                                             **
**                                                                  **
** GLOBALS        :none                                             **
**                                                                  **
** PARAMETERS     :*idum    serves as a seed value                  **
**                                                                  **
** PRECONDITIONS  :*idum must be negative on the first call.        **
**                                                                  **
** POSTCONDITIONS :*idum will be changed                            **
**                                                                  **
***C*F*E*************************************************************/
{
	long j;
	long k;
	static long idum2 = 123456789;
	static long iy = 0;
	static long iv[NTAB];
	double temp;

	if (*idum <= 0)
	{
		if (-(*idum) < 1) *idum = 1;
		else *idum = -(*idum);
		idum2 = (*idum);
		for (j = NTAB + 7; j >= 0; j--)
		{
			k = (*idum) / IQ1;
			*idum = IA1*(*idum - k*IQ1) - k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ1;
	*idum = IA1*(*idum - k*IQ1) - k*IR1;
	if (*idum < 0) *idum += IM1;
	k = idum2 / IQ2;
	idum2 = IA2*(idum2 - k*IQ2) - k*IR2;
	if (idum2 < 0) idum2 += IM2;
	j = iy / NDIV;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp = AM*iy) > RNMX) return RNMX;
	else return temp;

}/*------End of rnd_uni()--------------------------*/

int GetAlignedCircles(vector<Vec3f> circles){
	const int noOfCircles = circles.size();
	int noOfFinalCircles = 0;
	int **adjacency = new int*[noOfCircles];
	for (int i = 0; i < noOfCircles; i++)
		adjacency[i] = new int[noOfCircles];

	//Sum
	int* adjacencySum = new int[noOfCircles];
	//Cost
	int* centerCost = new int[noOfCircles];
	for (int i = 0; i < noOfCircles; i++){
		adjacencySum[i] = 0;
	}
	for (int i = 0; i < noOfCircles; i++){
		centerCost[i] = 0;
	}
	for (int i = 0; i < noOfCircles; i++){
		//cout << "Circle No " << i << " with center " << circles[i][1] << endl;
		for (int j = 0; j < noOfCircles; j++){
			if (i == j){
				adjacency[i][j] = 1;
			}
			else if (abs(cvRound(circles[i][1]) - cvRound(circles[j][1])) <= YTHRESHOLD) {
				adjacency[i][j] = 1;
			}
			else{
				adjacency[i][j] = 0;
			}
			//if (noOfCircles==3)
				//cout << adjacency[i][j] << ",";
		}
		//if (noOfCircles == 3)
			//cout << endl;
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
	int alignedCircles = 0;
	//Handle the special case of 2 circles
	if (noOfCircles%2==0){//even circles
		//check if the circles are divided in half
		if (max == noOfCircles/2)
			return noOfCircles/2;
		//check if all the circles are misaligned
		else if (noOfCircles/max == noOfCircles){
			return 0;
		}
		else{
			for (int i = 0; i < noOfCircles; i++){
				if (adjacencySum[i] == max){
					alignedCircles++;
				}
			}
		}
	}
	else{//odd circles
		//check if all the circles are misaligned
		if (noOfCircles / max == noOfCircles){
			//cout << "------------All the circles are misaligned-------------" << endl;
			return 0;
		}
		for (int i = 0; i < noOfCircles; i++){
			if (adjacencySum[i] == max){
				alignedCircles++;
			}
		}
	}

	return ( alignedCircles);

}
int GetAlignedRadii(vector<Vec3f> circles){
	int aligned = 1;
	if (circles.size() <= 1)
		return 1;
	vector<int> raddi (circles.size());
	for (int i = 0; i < circles.size(); i++){
		raddi[i] = (int)circles[i][2];
		//cout << "Radius is " << raddi[i] << endl;
	}
	sort(raddi.begin(), raddi.end());
	int diff = (raddi[circles.size() - 1] - raddi[0]);
	if ( diff<= RADIUSTHRES)
		aligned = 0;
	return aligned;
}
/*int GetAlignedRadii(vector<Vec3f> circles){
	const int noOfCircles = circles.size();
	int **adjacency = new int*[noOfCircles];
	for (int i = 0; i < noOfCircles; i++)
		adjacency[i] = new int[noOfCircles];

	//Sum
	int* adjacencySum = new int[noOfCircles];
	//Cost
	int* centerCost = new int[noOfCircles];
	for (int i = 0; i < noOfCircles; i++){
		adjacencySum[i] = 0;
	}
	for (int i = 0; i < noOfCircles; i++){
		centerCost[i] = 0;
	}
	for (int i = 0; i < noOfCircles; i++){
		cout << "Radius for circle " << i << " is ->> " << cvRound(circles[i][2]) << endl;
		for (int j = 0; j < noOfCircles; j++){
			if (i == j){
				adjacency[i][j] = 1;
			}
			else if (abs(cvRound(circles[i][2]) - cvRound(circles[j][2])) <= RADIUSTHRES) {
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
	int alignedRaddi = 0;
	for (int i = 0; i < noOfCircles; i++){
		if (adjacencySum[i] == max){
			alignedRaddi++;
		}
	}

	return (alignedRaddi);
}*/
double GetRadiusCost(int actual, int aligned){
	return actual - aligned;
}
int GetCentroidCost(int actual, int aligned){
	return actual - aligned;
}
int GetCenterDistance(vector<Vec3f> circles){
	int isClose = 0;
	for (int i = 0; i < circles.size(); i++){
		for (int j = i+1; j < circles.size(); j++){
			//cout << "Distance between centers of " << i << " and " << j << " circles is ->>" << abs(circles[i][0] - circles[j][0]) << endl;
			if (abs(circles[i][0] - circles[j][0]) < MINCENTERDISTANCE){
				isClose = 1;
				//cout << "Breaking " << endl;
				return isClose;
			}
		}
	}
	//cvWaitKey(0);
	return isClose;
}
vector<Vec3f> DE(Mat src, Mat src_gray, vector<Vec3f> circles, char in[], char out[], int mStrategy, int noOfStrategies, int imageNo)
{
	int glbMin = GLBMIN;
	reset++;
	//char  chr;             /* y/n choice variable                */
	char  *strat[] =      
	{
		"",
		"DE/best/1/exp",
		"DE/rand/1/exp",
		"DE/rand-to-best/1/exp",
		"DE/best/2/exp",
		"DE/rand/2/exp",
		"DE/best/1/bin",
		"DE/rand/1/bin",
		"DE/rand-to-best/1/bin",
		"DE/best/2/bin",
		"DE/rand/2/bin"
	};

	int   i, j, L, n;      /* counting variables                 */
	int   r1, r2, r3, r4;  /* placeholders for random indexes    */
	int   r5;              /* placeholders for random indexes    */
	int   D;               /* Dimension of parameter vector      */
	int   NP;              /* number of population members       */
	int   imin;            /* index to member with lowest energy */
	int   refresh;         /* refresh rate of screen output      */
	int   strategy;        /* choice parameter for screen output */
	int   gen, genmax, seed;

	long  nfeval;          /* number of function evaluations     */

	double trial_cost;      /* buffer variable                    */
	double inibound_h[] = { 4, 20, 200, 100 };      /* upper parameter bound              */
	//double inibound_l;      /* lower parameter bound              */
	double inibound_l[] = { 1, 10, 40, 20 };      /* lower parameter bound              */
	double tmp[MAXDIM], best[MAXDIM], bestit[MAXDIM]; /* members  */
	double cost[MAXPOP];    /* obj. funct. values                 */
	vector<Vec3f> costCircle[MAXPOP];
	double cvar;            /* computes the cost variance         */
	double cmean;           /* mean cost                          */
	double F, CR;            /* control variables of DE            */
	double cmin;            /* help variables                     */
	vector<Vec3f> bestSol;

	FILE  *fpin_ptr;
	FILE  *fpout_ptr;

	/*------Initializations----------------------------*/
	fflush(stdin);

	fpout_ptr = fopen(out, "r");          /* open output file for reading,    */
	/* to see whether it already exists */
	/*if (fpout_ptr != NULL)
	{
		printf("\nOutput file %s does already exist, \ntype y if you ", out);
		printf("want to overwrite it, \nanything else if you want to exit.\n");
		chr = (char)getchar();
		if ((chr != 'y') && (chr != 'Y'))
		{
			exit(1);
		}
		fclose(fpout_ptr);
	}*/

	printf("Arg1 %s ", in);
	printf("Arg2 %s ", out);
	/*-----Read input data------------------------------------------------*/

	fpin_ptr = fopen(in, "r");

	if (fpin_ptr == NULL)
	{
		printf("\nCannot open input file\n");
		pause("2");
		//pause("After");
		exit(1);                                 /* input file is necessary */
	}
	double a, b;
	fscanf(fpin_ptr, "%d", &strategy);       /*---choice of strategy-----------------*/
	fscanf(fpin_ptr, "%d", &genmax);         /*---maximum number of generations------*/
	fscanf(fpin_ptr, "%d", &refresh);        /*---output refresh cycle---------------*/
	fscanf(fpin_ptr, "%d", &D);              /*---number of parameters---------------*/
	fscanf(fpin_ptr, "%d", &NP);             /*---population size.-------------------*/
	fscanf(fpin_ptr, "%lf", &a);    /*---upper parameter bound for init-----*/
	fscanf(fpin_ptr, "%lf", &b);    /*---lower parameter bound for init-----*/
	fscanf(fpin_ptr, "%lf", &F);             /*---weight factor----------------------*/
	fscanf(fpin_ptr, "%lf", &CR);            /*---crossing over factor---------------*/
	fscanf(fpin_ptr, "%d", &seed);           /*---random seed------------------------*/


	fclose(fpin_ptr);
	//pause("2");
	/*-----Checking input variables for proper range----------------------------*/

	if (D > MAXDIM)
	{
		printf("\nError! D=%d > MAXDIM=%d\n", D, MAXDIM);
		exit(1);
	}
	if (D <= 0)
	{
		printf("\nError! D=%d, should be > 0\n", D);
		exit(1);
	}
	if (NP > MAXPOP)
	{
		printf("\nError! NP=%d > MAXPOP=%d\n", NP, MAXPOP);
		exit(1);
	}
	if (NP <= 0)
	{
		printf("\nError! NP=%d, should be > 0\n", NP);
		exit(1);
	}
	if ((CR < 0) || (CR > 1.0))
	{
		printf("\nError! CR=%f, should be ex [0,1]\n", CR);
		exit(1);
	}
	if (seed <= 0)
	{
		printf("\nError! seed=%d, should be > 0\n", seed);
		exit(1);
	}
	if (refresh <= 0)
	{
		printf("\nError! refresh=%d, should be > 0\n", refresh);
		exit(1);
	}
	if (genmax <= 0)
	{
		printf("\nError! genmax=%d, should be > 0\n", genmax);
		exit(1);
	}
	if ((strategy < 0) || (strategy > 10))
	{
		printf("\nError! strategy=%d, should be ex {1,2,3,4,5,6,7,8,9,10}\n", strategy);
		exit(1);
	}
	if (inibound_h < inibound_l)
	{
		printf("\nError! inibound_h=%f < inibound_l=%f\n", inibound_h, inibound_l);
		exit(1);
	}
	//pause("3");

	/*-----Open output file-----------------------------------------------*/

	fpout_ptr = fopen(out, "a+");  /* open output file for writing */

	if (fpout_ptr == NULL)
	{
		printf("\nCannot open output file\n");
		exit(1);
	}


	/*-----Initialize random number generator-----------------------------*/

	rnd_uni_init = -(long)seed;  /* initialization of rnd_uni() */
	nfeval = 0;  /* reset number of function evaluations */



	/*------Initialization------------------------------------------------*/
	/*------Right now this part is kept fairly simple and just generates--*/
	/*------random numbers in the range [-initfac, +initfac]. You might---*/
	/*------want to extend the init part such that you can initialize-----*/
	/*------each parameter separately.------------------------------------*/
	vector<Vec3f> deCircles;
	for (i = 0; i<NP; i++)
	{
		for (j = 0; j<D; j++) /* spread initial population members */
		{
			c[i][j] = inibound_l[j] + rnd_uni(&rnd_uni_init)*(inibound_h[j] - inibound_l[j]);
			if (c[i][j] < inibound_l[j])
				c[i][j] = inibound_l[j];
			if (c[i][j] > inibound_h[j])
				c[i][j] = inibound_h[j];
		}
		deCircles = evaluate(src, src_gray, D, c[i], &nfeval); /* obj. funct. value */
		//*************Model 1 start ************************//
		/*cost[i] = deCircles.size();
		if (cost[i] < (double)circles.size()){
			cost[i] = MAXCIRCLES;
		}
		else if (cost[i] == (double)circles.size()){
			bool match = checkCenteroidMatch(circles,deCircles);
		}
		else if (cost[i]>(double)circles.size() && cost[i] < (double)MAXCIRCLES){
			int noOfCircles = checkExtraCentroids(src,circles, deCircles);
			cout << "No of extra circles are " << noOfCircles<<endl;
			fflush(stdin);
			//pause("");
		}*/
		//*************Model 1 End ************************//

		//*************Model 2 start ************************//
		cost[i] = (double)deCircles.size();
		//cout << "Cost for solution No " << i << " before adding center cost is ->> " << cost[i] << endl;
		int alignedCircles = GetAlignedCircles(deCircles);
		cout << "Aligned circles are ->> " << alignedCircles << endl;

		cost[i] += (double)GetCentroidCost(deCircles.size(), alignedCircles);
		cost[i] += pow(MAXCIRCLES, 2) / (alignedCircles + EPSILON);
		cost[i] += pow(MAXCIRCLES, 2) * ((double)deCircles.size() - alignedCircles + 0.0);

		int alignedRaddi = GetAlignedRadii(deCircles);
		//cout << "Aligned raddi factor is  ->> " << alignedRaddi << endl;
		cost[i] += pow(MAXCIRCLES, 2) *(alignedRaddi*1.0);

		int isClose = GetCenterDistance(deCircles);
		//cout << "Min center distance factor is ->> " << isClose << endl;
		cost[i] += pow(MAXCIRCLES, 2) *(isClose*1.0);

		costCircle[i] = deCircles;
		//cout << "Cost for solution No " << i << " after adding center cost is ->> " << cost[i] << endl;
		//cout << "****************************************************************" << endl;
		//cvWaitKey(0);
	}
	cmin = cost[0];
	imin = 0;
	for (i = 1; i<NP; i++)
	{
		if (cost[i]<cmin)
		{
			cmin = cost[i];
			imin = i;
			//bestSol = deCircles;
		}
	}
	bestSol = costCircle[imin];
	///cout << "Best index is " << imin << endl;
	//cout << "Best is " << bestSol.size() << " with cost " << cmin << endl;
	//cout << "Best is " << costCircle[imin].size() << " with cost " << cmin << endl;
	//cvWaitKey(0);
	assignd(D, best, c[imin]);            /* save best member ever          */
	assignd(D, bestit, c[imin]);          /* save best member of generation */

	pold = &c; /* old population (generation G)   */
	pnew = &d; /* new population (generation G+1) */

	/*=======================================================================*/
	/*=========Iteration loop================================================*/
	/*=======================================================================*/

	gen = 0;                          /* generation counter reset */
	while ((gen < genmax) /*&& (kbhit() == 0)*/) /* remove comments if conio.h */
	{                                            /* is accepted by compiler    */
		gen++;
		imin = 0;

		for (i = 0; i<NP; i++)         /* Start of loop through ensemble  */
		{
			do                        /* Pick a random population member */
			{                         /* Endless loop for NP < 2 !!!     */
				r1 = (int)(rnd_uni(&rnd_uni_init)*NP);
			} while (r1 == i);

			do                        /* Pick a random population member */
			{                         /* Endless loop for NP < 3 !!!     */
				r2 = (int)(rnd_uni(&rnd_uni_init)*NP);
			} while ((r2 == i) || (r2 == r1));

			do                        /* Pick a random population member */
			{                         /* Endless loop for NP < 4 !!!     */
				r3 = (int)(rnd_uni(&rnd_uni_init)*NP);
			} while ((r3 == i) || (r3 == r1) || (r3 == r2));

			do                        /* Pick a random population member */
			{                         /* Endless loop for NP < 5 !!!     */
				r4 = (int)(rnd_uni(&rnd_uni_init)*NP);
			} while ((r4 == i) || (r4 == r1) || (r4 == r2) || (r4 == r3));

			do                        /* Pick a random population member */
			{                         /* Endless loop for NP < 6 !!!     */
				r5 = (int)(rnd_uni(&rnd_uni_init)*NP);
			} while ((r5 == i) || (r5 == r1) || (r5 == r2) || (r5 == r3) || (r5 == r4));


			/*=======Choice of strategy===============================================================*/
			/*=======We have tried to come up with a sensible naming-convention: DE/x/y/z=============*/
			/*=======DE :  stands for Differential Evolution==========================================*/
			/*=======x  :  a string which denotes the vector to be perturbed==========================*/
			/*=======y  :  number of difference vectors taken for perturbation of x===================*/
			/*=======z  :  crossover method (exp = exponential, bin = binomial)=======================*/
			/*                                                                                        */
			/*=======There are some simple rules which are worth following:===========================*/
			/*=======1)  F is usually between 0.5 and 1 (in rare cases > 1)===========================*/
			/*=======2)  CR is between 0 and 1 with 0., 0.3, 0.7 and 1. being worth to be tried first=*/
			/*=======3)  To start off NP = 10*D is a reasonable choice. Increase NP if misconvergence=*/
			/*           happens.                                                                     */
			/*=======4)  If you increase NP, F usually has to be decreased============================*/
			/*=======5)  When the DE/best... schemes fail DE/rand... usually works and vice versa=====*/


			/*=======EXPONENTIAL CROSSOVER============================================================*/

			/*-------DE/best/1/exp--------------------------------------------------------------------*/
			/*-------Our oldest strategy but still not bad. However, we have found several------------*/
			/*-------optimization problems where misconvergence occurs.-------------------------------*/
			if (mStrategy > 0)
				strategy = mStrategy;
			//cout << " Switching to strategy " << mStrategy << endl;
			//cvWaitKey(0);
			if (strategy == 1) /* strategy DE0 (not in our paper) */
			{
				assignd(D, tmp, (*pold)[i]);
				n = (int)(rnd_uni(&rnd_uni_init)*D);
				L = 0;
				do
				{
					tmp[n] = bestit[n] + F*((*pold)[r2][n] - (*pold)[r3][n]);
					n = (n + 1) % D;
					L++;
				} while ((rnd_uni(&rnd_uni_init) < CR) && (L < D));
			}
			/*-------DE/rand/1/exp-------------------------------------------------------------------*/
			/*-------This is one of my favourite strategies. It works especially well when the-------*/
			/*-------"bestit[]"-schemes experience misconvergence. Try e.g. F=0.7 and CR=0.5---------*/
			/*-------as a first guess.---------------------------------------------------------------*/
			else if (strategy == 2) /* strategy DE1 in the techreport */
			{
				assignd(D, tmp, (*pold)[i]);
				n = (int)(rnd_uni(&rnd_uni_init)*D);
				L = 0;
				do
				{
					tmp[n] = (*pold)[r1][n] + F*((*pold)[r2][n] - (*pold)[r3][n]);
					n = (n + 1) % D;
					L++;
				} while ((rnd_uni(&rnd_uni_init) < CR) && (L < D));
			}
			/*-------DE/rand-to-best/1/exp-----------------------------------------------------------*/
			/*-------This strategy seems to be one of the best strategies. Try F=0.85 and CR=1.------*/
			/*-------If you get misconvergence try to increase NP. If this doesn't help you----------*/
			/*-------should play around with all three control variables.----------------------------*/
			else if (strategy == 3) /* similiar to DE2 but generally better */
			{
				assignd(D, tmp, (*pold)[i]);
				n = (int)(rnd_uni(&rnd_uni_init)*D);
				L = 0;
				do
				{
					tmp[n] = tmp[n] + F*(bestit[n] - tmp[n]) + F*((*pold)[r1][n] - (*pold)[r2][n]);
					n = (n + 1) % D;
					L++;
				} while ((rnd_uni(&rnd_uni_init) < CR) && (L < D));
			}
			/*-------DE/best/2/exp is another powerful strategy worth trying--------------------------*/
			else if (strategy == 4)
			{
				assignd(D, tmp, (*pold)[i]);
				n = (int)(rnd_uni(&rnd_uni_init)*D);
				L = 0;
				do
				{
					tmp[n] = bestit[n] +
						((*pold)[r1][n] + (*pold)[r2][n] - (*pold)[r3][n] - (*pold)[r4][n])*F;
					n = (n + 1) % D;
					L++;
				} while ((rnd_uni(&rnd_uni_init) < CR) && (L < D));
			}
			/*-------DE/rand/2/exp seems to be a robust optimizer for many functions-------------------*/
			else if (strategy == 5)
			{
				assignd(D, tmp, (*pold)[i]);
				n = (int)(rnd_uni(&rnd_uni_init)*D);
				L = 0;
				do
				{
					tmp[n] = (*pold)[r5][n] +
						((*pold)[r1][n] + (*pold)[r2][n] - (*pold)[r3][n] - (*pold)[r4][n])*F;
					n = (n + 1) % D;
					L++;
				} while ((rnd_uni(&rnd_uni_init) < CR) && (L < D));
			}

			/*=======Essentially same strategies but BINOMIAL CROSSOVER===============================*/

			/*-------DE/best/1/bin--------------------------------------------------------------------*/
			else if (strategy == 6)
			{
				assignd(D, tmp, (*pold)[i]);
				n = (int)(rnd_uni(&rnd_uni_init)*D);
				for (L = 0; L<D; L++) /* perform D binomial trials */
				{
					if ((rnd_uni(&rnd_uni_init) < CR) || L == (D - 1)) /* change at least one parameter */
					{
						tmp[n] = bestit[n] + F*((*pold)[r2][n] - (*pold)[r3][n]);
					}
					n = (n + 1) % D;
				}
			}
			/*-------DE/rand/1/bin-------------------------------------------------------------------*/
			else if (strategy == 7)
			{
				assignd(D, tmp, (*pold)[i]);
				n = (int)(rnd_uni(&rnd_uni_init)*D);
				for (L = 0; L<D; L++) /* perform D binomial trials */
				{
					if ((rnd_uni(&rnd_uni_init) < CR) || L == (D - 1)) /* change at least one parameter */
					{
						tmp[n] = (*pold)[r1][n] + F*((*pold)[r2][n] - (*pold)[r3][n]);
					}
					n = (n + 1) % D;
				}
			}
			/*-------DE/rand-to-best/1/bin-----------------------------------------------------------*/
			else if (strategy == 8)
			{
				assignd(D, tmp, (*pold)[i]);
				n = (int)(rnd_uni(&rnd_uni_init)*D);
				for (L = 0; L<D; L++) /* perform D binomial trials */
				{
					if ((rnd_uni(&rnd_uni_init) < CR) || L == (D - 1)) /* change at least one parameter */
					{
						tmp[n] = tmp[n] + F*(bestit[n] - tmp[n]) + F*((*pold)[r1][n] - (*pold)[r2][n]);
					}
					n = (n + 1) % D;
				}
			}
			/*-------DE/best/2/bin--------------------------------------------------------------------*/
			else if (strategy == 9)
			{
				assignd(D, tmp, (*pold)[i]);
				n = (int)(rnd_uni(&rnd_uni_init)*D);
				for (L = 0; L<D; L++) /* perform D binomial trials */
				{
					if ((rnd_uni(&rnd_uni_init) < CR) || L == (D - 1)) /* change at least one parameter */
					{
						tmp[n] = bestit[n] +
							((*pold)[r1][n] + (*pold)[r2][n] - (*pold)[r3][n] - (*pold)[r4][n])*F;
					}
					n = (n + 1) % D;
				}
			}
			/*-------DE/rand/2/bin--------------------------------------------------------------------*/
			else
			{
				assignd(D, tmp, (*pold)[i]);
				n = (int)(rnd_uni(&rnd_uni_init)*D);
				for (L = 0; L<D; L++) /* perform D binomial trials */
				{
					if ((rnd_uni(&rnd_uni_init) < CR) || L == (D - 1)) /* change at least one parameter */
					{
						tmp[n] = (*pold)[r5][n] +
							((*pold)[r1][n] + (*pold)[r2][n] - (*pold)[r3][n] - (*pold)[r4][n])*F;
					}
					n = (n + 1) % D;
				}
			}


			/*=======Trial mutation now in tmp[]. Test how good this choice really was.==================*/
			deCircles = evaluate(src, src_gray, D, tmp, &nfeval);  /* Evaluate new vector in tmp[] */
			trial_cost = (double)deCircles.size();
			//cout << "No. of circles detected are ->> " << trial_cost << endl;

			int alignedCircles = GetAlignedCircles(deCircles);
			//cout << "Aligned circles are ->> " << alignedCircles << endl;
			trial_cost += pow(MAXCIRCLES, 2) / (alignedCircles + EPSILON);
			trial_cost += pow(MAXCIRCLES, 2) * (((double)deCircles.size() - alignedCircles) + 0.0);
			trial_cost += (double)GetCentroidCost(deCircles.size(), alignedCircles);

			int alignedRaddi = GetAlignedRadii(deCircles);
			//cout << "Aligned raddi factor is ->> " << alignedRaddi << endl;
			//trial_cost += GetRadiusCost(deCircles.size(), alignedRaddi);
			trial_cost += pow(MAXCIRCLES, 2) *(alignedRaddi*1.0);
			//trial_cost += pow(MAXCIRCLES, 2) * (((double)deCircles.size() - alignedRaddi) + 0.0);
			
			int isClose = GetCenterDistance(deCircles);
			//cout << "Min center distance factor is ->> " << isClose << endl;
			trial_cost += pow(MAXCIRCLES, 2) *(isClose*1.0);

			//commented for Model 2
			/*if (trial_cost < (double)circles.size()){
				trial_cost = MAXCIRCLES;
			}*/
			/*if (trial_cost < glbMin){
				trial_cost = MAXCIRCLES;
			}*/
			//cout << "Trial cost after center cost addition is " << trial_cost << endl;
			//cout << "***********************************************************" << endl;
			double alignmentCost = 0;
			if (trial_cost <= cost[i])   /* improved objective function value ? */
			{
				cost[i] = trial_cost;
				costCircle[i] = deCircles;
				assignd(D, (*pnew)[i], tmp);
				if (trial_cost<cmin)          /* Was this a new minimum? */
				{                               /* if so...*/
					cmin = trial_cost;           /* reset cmin to new low...*/
					imin = i;
					assignd(D, best, tmp);
					costCircle[i] = deCircles;
					//cout << "Best is " << deCircles.size() << " with cost " << cmin << endl;
					//alignmentCost = 
					bestSol = deCircles;
					//cvWaitKey(0);
				}
			}
			else
			{
				assignd(D, (*pnew)[i], (*pold)[i]); /* replace target with old value */
			}
		}   /* End mutation loop through pop. */

		assignd(D, bestit, best);  /* Save best population member of current iteration */

		/* swap population arrays. New generation becomes old one */

		pswap = pold;
		pold = pnew;
		pnew = pswap;

		/*----Compute the energy variance (just for monitoring purposes)-----------*/

		cmean = 0.;          /* compute the mean value first */
		for (j = 0; j<NP; j++)
		{
			cmean += cost[j];
		}
		cmean = cmean / NP;

		cvar = 0.;           /* now the variance              */
		for (j = 0; j<NP; j++)
		{
			cvar += (cost[j] - cmean)*(cost[j] - cmean);
		}
		cvar = cvar / (NP - 1);


		/*----Output part----------------------------------------------------------*/

		if (gen%refresh == 1)   /* display after every refresh generations */
		{ /* ABORT works only if conio.h is accepted by your compiler */
			printf("\n\n                         PRESS ANY KEY TO ABORT");
			printf("\n\n\n Best-so-far cost funct. value=%-15.10g\n", cmin);
			for (j = 0; j<D; j++)
			{
				printf("\n best[%d]=%-15.10g", j, best[j]);
			}
			printf("\n\n Generation=%d  NFEs=%ld   Strategy: %s    ", gen, nfeval, strat[strategy]);
			printf("\n NP=%d    F=%-4.2g    CR=%-4.2g   cost-variance=%-10.5g\n",
				NP, F, CR, cvar);
		}

		//fprintf(fpout_ptr, "%ld   %-15.10g\n", nfeval, cmin);
	}
	/*=========End of iteration loop=========================================*/
	cout << "Best is " << bestSol.size() << endl;
	cout << "Axles for image " << imageNo<< " are " << axles[imageNo] << endl;
	//cvWaitKey(0);
	int res = 0;
	if (bestSol.size() == axles[imageNo]){
		res = 1;
	}
	else{
		res = 0;
	}

	/*-------Final output in file-------------------------------------------*/
	fprintf(fpout_ptr, "%f|%d", cmin,res);
	

	//for (j = 0; j<D; j++)
	//{
		//fprintf(fpout_ptr, "\n best[%d]=%-15.10g", j, best[j]);
	//}
	//fprintf(fpout_ptr, "\n\n Generation=%d  NFEs=%ld   Strategy: %s    ", gen, nfeval, strat[strategy]);
	//fprintf(fpout_ptr, "\n NP=%d    F=%-4.2g    CR=%-4.2g    cost-variance=%-10.5g\n",
		//NP, F, CR, cvar);
	if (reset % noOfStrategies == 0){
		fprintf(fpout_ptr,"\n");
		reset = 0;
	}
	fclose(fpout_ptr);
	/*cout << "No of circles found are ->>> " << bestSol.size() << " with cost ->> "<< cmin<<endl;
	for (int i = 0; i < bestSol.size(); i++){
		cout << "******************************" << endl;
		cout << " x ->> " << bestSol[i][0] << endl;
		cout << " y ->> " << bestSol[i][1] << endl;
		cout << " z ->> " << bestSol[i][2] << endl;
		
	}*/
	return bestSol;
}

/*-----------End of main()------------------------------------------*/





