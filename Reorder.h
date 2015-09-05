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

	Parameter* para;

	MatrixLBFGS weights;
	MatrixLBFGS weights_b;
	
	MatrixLBFGS outputLayer;
	MatrixLBFGS softmaxLayer;

	MatrixLBFGS delWeight;
	MatrixLBFGS delWeight_b;

	lbfgsfloatval_t decay();
	int getRMWeightSize();
	void updateWeights(const lbfgsfloatval_t* x);
	void softmax();
	void trainRM(MatrixLBFGS y, bool isSoftmax);
	ReorderModel(Parameter* para, RAE* rae);
	void getData(string bp1, string bp2);
	ReorderModel* copy();
	~ReorderModel()
	{
		delete rae;
		delete rae2;
		delete rae1;
	}
};

#endif
