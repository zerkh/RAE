#include <iostream>
#include <ctime>
#include "Vec.h"
#include "Domain.h"
#include "Parameter.h"
#include "WordVec.h"

using namespace std;

int main()
{
	double start, end;

	start = clock();	
	Parameter* para = new Parameter("Para.ds");
	end = clock();
	cout << "The time of reading parameter is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	start = clock();
	Domain* domain = new Domain(para, "Education");
	domain->loadTrainingData();
	domain->training();
	end = clock();
	cout << "The time of processing Education is " << (end-start)/CLOCKS_PER_SEC << endl <<  endl;

	return 0;
}