#ifndef DOMAIN_H
#define DOMAIN_H

#include "Reorder.h"
#include "Util.h"

class Domain
{
public:
	ReorderModel* srcRM;
	ReorderModel* tgtRM;
	vector<pair<int, map<string, string> > > trainingData;
	string dataFile;
	int iterTime;
	string domainName;
	ofstream out, srcOut, tgtOut, srcWLog, tgtWLog;

public:
	Domain(Parameter* para, string domainName, WordVec* srcWords, WordVec* tgtWords);
	void loadTrainingData();
	void training();
	void upData();
	double loss(int ind);
	void test();
	void logWeights();
	void loadTestingData();
	void loadWeights();
};

#endif // !DOMAIN_H
