#ifndef REORDER_H
#define REORDER_H

#include "Util.h"
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

	MatrixXd weights;
	MatrixXd weights_b;
	
	MatrixXd outputLayer;
	MatrixXd softmaxLayer;

	MatrixXd delWeight;
	MatrixXd delWeight_b;

	double decay();
	void softmax();
	void trainRM(MatrixXd y, bool isSoftmax);
	ReorderModel(Parameter* para, WordVec* words);
	void getData(string bp1, string bp2);
};

#endif
