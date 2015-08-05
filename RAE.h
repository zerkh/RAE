#ifndef RAE_H
#define RAE_H

#include <fstream>
#include "WordVec.h"
#include "Parameter.h"
#include "RAETree.h"
#include <vector>
#include <sstream>
#include <map>

class RAE
{
private:
	WordVec* words;
	int vecSize;
	
public:
	Tree* RAETree;
	MatrixXd weights1;
	MatrixXd weights_b1;
	MatrixXd weights2;
	MatrixXd weights_b2;

	MatrixXd delWeight1;
	MatrixXd delWeight1_b;
	MatrixXd delWeight2;
	MatrixXd delWeight2_b;

	RAE(Parameter* para, WordVec* words);
	void showWeights();
	void logWeights(Parameter* para);
	int getVecSize();
	void buildTree(string bp);
	void trainRecError();
	double loss();
	double decay();
	RAE* copy();
	RAE(int size);
	~RAE();
};

#endif
