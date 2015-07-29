#ifndef DOMAIN_H
#define DOMAIN_H

#include "Reorder.h"
#include <fstream>

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
	Domain(Parameter* para, string domainName);
	void loadTrainingData();
	void training();
	void upData();
	double loss(int ind);
	void test();
	void logWeights();
};

#endif // !DOMAIN_H
