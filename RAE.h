#ifndef RAE_H
#define RAE_H

#include <fstream>
#include "Vec.h"
#include "WordVec.h"
#include "Parameter.h"
#include <map>

typedef pair<string, Vector*> p_StringVec;

class RAE
{
private:
	Vector* weights1;
	Vector* weights_b1;
	Vector* weights2;
	Vector* weights_b2;

	WordVec* words;
	int vecSize;

	double loss(Vector* inputLayer, Vector* recLayer);
public:
	RAE(Parameter* para, WordVec* words);
	void showWeights();
	void logWeights(Parameter* para);
	int getVecSize();
	p_StringVec getStringVec(string word1 ,string word2);
	void trainRAE();
};

#endif