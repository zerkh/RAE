#include <iostream>
#include <ctime>
#include "Vec.h"
#include "Parameter.h"

using namespace std;

int main()
{
	double start, end;

	start = clock();	
	Parameter* para = new Parameter("Para.ds");
	end = clock();
	cout << "The time of reading parameter is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	start = clock();

	end = clock();
}