#ifndef MIXEDDOMAIN
#define MIXEDDOMAIN

#include "Util.h"
#include "Domain.h"
#include "MutiThreading.h"
#include "ThreadPara.h"

void train( worker_arg_t *arg );
void test( worker_arg_t *arg );
void buildArct(worker_arg_t* arg);

class MixedDomain
{
public:
	int amountOfDomains;
	vector<Domain*> domains;
	Domain* mixedDomain;
	RAE* srcRAE;
	RAE* tgtRAE;
	lbfgsfloatval_t* x;
	Parameter* para;
	worker_arg_t *wargs;
	vector<pair<string, string> > unlabelData;

	MixedDomain()
	{
		amountOfDomains = 0;
		srcRAE = NULL;
		tgtRAE = NULL;
		wargs = NULL;
	};

	MixedDomain(Parameter* para, vector<Domain*>& domains, RAE* srcRAE, RAE* tgtRAE);

	lbfgsfloatval_t _training(lbfgsfloatval_t* g);
	void getUnlabelData(string filename);
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

	void training();

	void testing();

	void mixedTesting();

	vector<pair<int, map<string, string> > > getTestData();
};


static lbfgsfloatval_t evaluate(
	void *instance,
	const lbfgsfloatval_t *x,
	lbfgsfloatval_t *g,
	const int n,
	const lbfgsfloatval_t step
	)
{
	return reinterpret_cast<MixedDomain*>(instance)->_evaluate(x, g, n, step);
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
	return reinterpret_cast<MixedDomain*>(instance)->_progress(x, g, fx, xnorm, gnorm, step, n, k, ls);
}
#endif