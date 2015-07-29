#ifndef RAE_H
#define RAE_H

#include <fstream>
#include "Vec.h"
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
	Vector* weights1;
	Vector* weights_b1;
	Vector* weights2;
	Vector* weights_b2;

	Vector* delWeight1;
	Vector* delWeight1_b;
	Vector* delWeight2;
	Vector* delWeight2_b;

	RAE(Parameter* para, WordVec* words);
	void showWeights();
	void logWeights(Parameter* para);
	int getVecSize();
	Tree* buildTree(string bp);
	void trainRecError();
	double loss();
	double decay();
	RAE* copy();
	RAE();
	~RAE();
};

#endif