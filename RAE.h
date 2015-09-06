#ifndef RAE_H
#define RAE_H

#include "WordVec.h"
#include "Parameter.h"
#include "RAETree.h"
#include "Util.h"

class RAE
{
private:
	WordVec* words;
	int vecSize;
	int iterTimes;
	Parameter* para;

public:
	Tree* RAETree;
	lbfgsfloatval_t* x;

	MatrixLBFGS weights1;
	MatrixLBFGS weights_b1;
	MatrixLBFGS weights2;
	MatrixLBFGS weights_b2;

	MatrixLBFGS delWeight1;
	MatrixLBFGS delWeight1_b;
	MatrixLBFGS delWeight2;
	MatrixLBFGS delWeight2_b;

	vector<string> trainingData;
	int RAEType;

	RAE(Parameter* para, WordVec* words, int RAEType);
	void showWeights();
	void logWeights(Parameter* para);
	int getVecSize();
	void buildTree(string bp);
	void trainRecError();
	void loadTrainingData();
	lbfgsfloatval_t loss();
	lbfgsfloatval_t decay();
	int getRAEWeightSize();
	void delToZero();
	RAE* copy();
	RAE(int size);
	~RAE();
	void updateWeights(const lbfgsfloatval_t* x);
	void update(lbfgsfloatval_t* g);
	void loadWeights(Parameter* para);
};
#endif
