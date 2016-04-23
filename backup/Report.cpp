#include "opencv/highgui.h"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;
using namespace cv;

ofstream outClass;
ifstream inFile;

vector<string> vehicles = { "Car", "SUV", "Pick Up", "Truck", "Truck 1", "Truck 2", "Truck 3" };

void GenerateReport(){
	inFile.open("data/report.txt");
	outClass.open("data/FinalReport.txt", 'w');
	string line = "";
	string tmp = "";
	int times = 0;
	for (int i = 0; i < vehicles.size(); i++){
		line = vehicles[i];
		cout << "Vechicle is " << line << endl;
		while (getline(inFile,tmp)){
			cout << "comparing " << line << " and " << tmp << endl;
			if (line.compare(tmp) == 0){
				cout << line << " found " << times + 1 << " times" << endl;
				times++;
			}
		}
		outClass << line << " - " << times << endl;
		times = 0;
		inFile.clear();
		inFile.seekg(0,inFile.beg);
	}
	outClass.close();
	inFile.close();
}

void main(){
	GenerateReport();
	int s = 0;
	cin >> s; 
}