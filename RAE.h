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
	void training();
	lbfgsfloatval_t _training(lbfgsfloatval_t* g);
	lbfgsfloatval_t _evaluate(const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step);
	int _progress(const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, 
		const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
		int n, int k, int ls);
	void logWeights(Parameter* para);
	int getVecSize();
	void buildTree(string bp);
	void trainRecError();
	void loadTrainingData();
	lbfgsfloatval_t loss();
	lbfgsfloatval_t decay();
	int getRAEWeightSize();
	RAE* copy();
	RAE(int size);
	~RAE();
	void updateWeights(const lbfgsfloatval_t* x);
	void update(lbfgsfloatval_t* g);
	void loadWeights(Parameter* para)
};

namespace RAELBFGS
{
	static lbfgsfloatval_t evaluate(
		void *instance,
		const lbfgsfloatval_t *x,
		lbfgsfloatval_t *g,
		const int n,
		const lbfgsfloatval_t step
		)
	{
		return reinterpret_cast<RAE*>(instance)->_evaluate(x, g, n, step);
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
		return reinterpret_cast<RAE*>(instance)->_progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
	}

	static void* deepThread(void* args);
}
#endif
