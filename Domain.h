#ifndef DOMAIN_H
#define DOMAIN_H

#include "Reorder.h"
#include "Util.h"

class Domain
{
public:
	ReorderModel* srcRM;
	//ReorderModel* tgtRM;
	RAE* srcRAE;
	//RAE* tgtRAE;
	vector<pair<int, map<string, string> > > trainingData;
	string dataFile;
	int iterTime;
	string domainName;
	ofstream out, srcOut, tgtOut, srcWLog, tgtWLog;
	Parameter* para;

public:
	//Domain(Parameter* para, string domainName, RAE* srcRAE, RAE* tgtRAE);
	Domain(Parameter* para, string domainName, RAE* srcRAE);
	void loadTrainingData();
	pair<lbfgsfloatval_t, lbfgsfloatval_t> training(lbfgsfloatval_t* g_RM, lbfgsfloatval_t* g_RAE);
	void update(lbfgsfloatval_t* g_RM, lbfgsfloatval_t* g_RAE);
	int getWeightsSize();
	lbfgsfloatval_t loss(int ind);
	void test();
	void logWeights();
	void loadTestingData();
	Domain* copy();
	void loadWeights();
	~Domain()
	{
		delete srcRM;
		//delete tgtRM;
	}
};


static void copyDelweights(RAE* rae1, RAE* rae2)
{
	rae1->delWeight1 += rae2->delWeight1;
	rae1->delWeight1_b += rae2->delWeight1_b;
	rae1->delWeight2 += rae2->delWeight2;
	rae1->delWeight2_b += rae2->delWeight2_b;
}

#endif // !DOMAIN_H
