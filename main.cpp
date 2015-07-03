#include <iostream>
#include <ctime>
#include "Vec.h"
#include "RAE.h"
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
	WordVec* words = new WordVec();
	words->readFile(para);
	end = clock();
	cout << "The time of reading  word vector is " << (end-start)/CLOCKS_PER_SEC << endl <<  endl;

	start = clock();
	
	end = clock();
}