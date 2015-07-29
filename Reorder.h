#ifndef REORDER_H
#define REORDER_H
#include "RAE.h"
#include "Parameter.h"
#include "WordVec.h"
#include <cmath>

using namespace std;

class ReorderModel
{
private:
	int vecSize;

public:
	RAE* rae1;
	RAE* rae2;
	RAE* rae;

	Vector* weights;
	Vector* weights_b;
	
	Vector* outputLayer;
	Vector* softmaxLayer;

	Vector* delWeight;
	Vector* delWeight_b;

	double decay();
	void softmax();
	void trainRM(Vector* y, bool isSoftmax);
	ReorderModel(Parameter* para, WordVec* words);
	void getData(string bp1, string bp2);
}

#endif