#include "MixedDomain.h"

static void* deepThread(void* arg)
{
	ThreadPara* threadpara = (ThreadPara*)arg;

	threadpara->lossVal = threadpara->d->training(threadpara->g_RM, threadpara->g_RAE);

	pthread_exit(NULL);
}

MixedDomain::MixedDomain(Parameter* para, vector<Domain*>& domains, RAE* srcRAE, RAE* tgtRAE)
{
	this->domains = domains;
	this->srcRAE = srcRAE;
	this->tgtRAE = tgtRAE;
	this->para = para;
	this->amountOfDomains = domains.size();

	// thread_num = 1
	int thread_num = atoi( para->getPara("THREAD_NUM").c_str() );
	vector<string> v_domains;

	string domainLine = para->getPara("DomainList");
	string domainName;
	stringstream ss(domainLine);
	int count = 0;
	while(ss >> domainName)
	{
		if(count == thread_num)
		{
			break;
		}
		count++;
		v_domains.push_back(domainName);
	}

	//初始化
	wargs = new worker_arg_t[thread_num];

	//分发文件
	for (int wid = 0; wid < thread_num; wid++)
	{
		wargs[wid].m_id = wid;
		wargs[wid].domainName = v_domains[wid];
		wargs[wid].domain = domains[wid];
	}
}

void MixedDomain::training()
{
	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);
	param.max_iterations = atoi(para->getPara("IterationTime").c_str());
	int vecSize = atoi(para->getPara("WordVecSize").c_str());

	x = lbfgs_malloc((vecSize*2*2 + 2)*2*domains.size() + srcRAE->getRAEWeightSize()*2);
	Map<MatrixLBFGS>(x, (vecSize*2*2 + 2)*2*domains.size() + srcRAE->getRAEWeightSize()*2, 1).setRandom();

	lbfgsfloatval_t fx = 0;
	int ret = 0;
	ret = lbfgs( (vecSize*2*2 + 2)*2*domains.size() + srcRAE->getRAEWeightSize()*2, x, &fx, evaluate, progress, this, &param);

	cout << "L-BFGS optimization terminated with status code = " << ret << endl;
	cout << " fx = " << fx << endl;

	//记录权重
	for(int i = 0; i < domains.size(); i++)
	{
		domains[i]->logWeights();
	}

	lbfgs_free(x);
}

lbfgsfloatval_t MixedDomain::_evaluate(const lbfgsfloatval_t* x,
								  lbfgsfloatval_t* g,
								  const int n,
								  const lbfgsfloatval_t step)
{
	lbfgsfloatval_t fx = 0;

	srcRAE->updateWeights(x);
	tgtRAE->updateWeights(x + srcRAE->getRAEWeightSize());
	
	for(int i = 0; i < amountOfDomains; i++)
	{
		domains[i]->srcRM->rae = srcRAE->copy();
		domains[i]->tgtRM->rae = tgtRAE->copy();
		domains[i]->srcRM->updateWeights(x + srcRAE->getRAEWeightSize()*2 + i*2*domains[i]->srcRM->getRMWeightSize());
		domains[i]->tgtRM->updateWeights(x + srcRAE->getRAEWeightSize()*2 + (i*2+1)*domains[i]->srcRM->getRMWeightSize());
		wargs[i].g_RAE = g;
		wargs[i].g_RM = g + srcRAE->getRAEWeightSize()*2 + i*2*domains[i]->srcRM->getRMWeightSize();
	}

	Start_Workers(train, wargs, amountOfDomains);

	int count = 0;
	for(int i = 0; i < amountOfDomains; i++)
	{
		count += wargs[i].domain->trainingData.size();
		fx += wargs[i].error;
	}

	fx /= amountOfDomains;

	for(int i = 0; i < srcRAE->getRAEWeightSize()*2; i++)
	{
		g[i] /= count;
	}

	return fx;
}

int MixedDomain::_progress(const lbfgsfloatval_t *x,
					  const lbfgsfloatval_t *g,
					  const lbfgsfloatval_t fx,
					  const lbfgsfloatval_t xnorm,
					  const lbfgsfloatval_t gnorm,
					  const lbfgsfloatval_t step,
					  int n,
					  int k,
					  int ls)
{
	cout << "Iteration: " << k << endl;
	cout << "Loss Value: " << fx << endl;

	return 0;
}

void MixedDomain::testing()
{
	Start_Workers(test, wargs, amountOfDomains);
}

void train(worker_arg_t* arg)
{
	lbfgsfloatval_t start, end;

	Domain* d = arg->domain;
	cout << "Processing " << d->domainName << "......" << endl << endl;

	cout << "Loading " + d->domainName + " training data..." << endl << endl;
	start = clock();
	d->loadTrainingData();
	end = clock();
	cout << "The time of loading " + d->domainName + " training data is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	int RMThreadNum = atoi(d->para->getPara("RMThreadNum").c_str());
	ThreadPara* threadpara = new ThreadPara[RMThreadNum];
	int batchsize = d->trainingData.size() / RMThreadNum;

	for(int i = 0; i < RMThreadNum; i++)
	{
		threadpara[i].d = d->copy();
		threadpara[i].g_RAE = arg->g_RAE;
		threadpara[i].g_RM = arg->g_RM;
		if(i == RMThreadNum-1)
		{
			threadpara[i].d->trainingData.assign(d->trainingData.begin()+i*batchsize, d->trainingData.end());
			threadpara[i].instance_num = d->trainingData.size()%batchsize;
		}
		else
		{
			threadpara[i].d->trainingData.assign(d->trainingData.begin()+i*batchsize, d->trainingData.begin()+(i+1)*batchsize);
			threadpara[i].instance_num = batchsize;
		}
	}
	pthread_t* pt = new pthread_t[RMThreadNum];
	for (int a = 0; a < RMThreadNum; a++) pthread_create(&pt[a], NULL, deepThread, (void *)(threadpara + a));
	for (int a = 0; a < RMThreadNum; a++) pthread_join(pt[a], NULL);

	lbfgsfloatval_t fx = 0;
	for(int i = 0; i < RMThreadNum; i++)
	{
		fx += threadpara[i].lossVal;
	}

	fx /= d->trainingData.size();
	for(int elem = 0; elem < d->getWeightsSize(); elem++)
	{
		arg->g_RM[elem] /= d->trainingData.size();
	}

	delete pt;
	pt = NULL;

	for(int i  = 0; i < RMThreadNum; i++)
	{
		delete threadpara[i].d;
	}
	delete threadpara;
	threadpara = NULL;

	arg->error = fx;
}

void test(worker_arg_t* arg)
{
	lbfgsfloatval_t start, end;

	Domain* d = arg->domain;
	cout << "Processing " << d->domainName << "......" << endl << endl;

	cout << "Loading " + d->domainName + " testing data..." << endl << endl;
	start = clock();
	d->loadTestingData();
	end = clock();
	cout << "The time of loading " + d->domainName + " testing data is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	cout << "Starting testing " + d->domainName + "..." << endl << endl;
	start = clock();
	d->test();
	end = clock();
	cout << "The time of testing " + d->domainName + " is " << (end-start)/CLOCKS_PER_SEC << endl << endl;
}