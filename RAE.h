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
	
public:
	Tree* RAETree;
	MatrixLBFGS weights1;
	MatrixLBFGS weights_b1;
	MatrixLBFGS weights2;
	MatrixLBFGS weights_b2;

	MatrixLBFGS delWeight1;
	MatrixLBFGS delWeight1_b;
	MatrixLBFGS delWeight2;
	MatrixLBFGS delWeight2_b;

	RAE(Parameter* para, WordVec* words);
	void showWeights();
	void logWeights(Parameter* para);
	int getVecSize();
	void buildTree(string bp);
	void trainRecError();
	lbfgsfloatval_t loss();
	lbfgsfloatval_t decay();
	int getRAEWeightSize();
	RAE* copy();
	RAE(int size);
	~RAE();
};

#endif
