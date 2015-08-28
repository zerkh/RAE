#ifndef DOMAIN_H
#define DOMAIN_H

#include "Reorder.h"
#include "Util.h"

static lbfgsfloatval_t evaluate(
	void *instance,
	const lbfgsfloatval_t *x,
	lbfgsfloatval_t *g,
	const int n,
	const lbfgsfloatval_t step
	)
{
	return reinterpret_cast<Domain*>(instance)->_evaluate(x, g, n, step);
}

static int progress(
	void *instance,
	const lbfgsfloatval_t *x,
	const lbfgsfloatval_t *g,
	const lbfgsfloatval_t fx,
	const lbfgsfloatval_t xnorm,
	const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step,
	int n,
	int k,
	int ls
	)
{
	return reinterpret_cast<Domain*>(instance)->_progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
}

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
	lbfgsfloatval_t* x;

public:
	Domain(Parameter* para, string domainName, WordVec* srcWords, WordVec* tgtWords);
	void loadTrainingData();
	void training();
	void upData(lbfgsfloatval_t* g);
	int getWeightsSize();
	lbfgsfloatval_t loss(int ind);
	void test();
	void logWeights();
	void loadTestingData();
	void loadWeights();
	lbfgsfloatval_t _training(lbfgsfloatval_t* g);
	lbfgsfloatval_t _evaluate(const lbfgsfloatval_t* x,
		lbfgsfloatval_t* g,
		const int n,
		const lbfgsfloatval_t step);
	int _progress(const lbfgsfloatval_t *x,
		const lbfgsfloatval_t *g,
		const lbfgsfloatval_t fx,
		const lbfgsfloatval_t xnorm,
		const lbfgsfloatval_t gnorm,
		const lbfgsfloatval_t step,
		int n,
		int k,
		int ls);
};

#endif // !DOMAIN_H
