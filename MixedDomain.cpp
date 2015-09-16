#include "MixedDomain.h"

static void* deepThread(void* arg)
{
	ThreadPara* threadpara = (ThreadPara*)arg;

	threadpara->lossVal = threadpara->d->training(threadpara->g_RM, threadpara->g_RAE);

	pthread_exit(NULL);
}

static void* srcUnlabelThread(void* arg)
{
	UnlabelThreadPara* threadpara = (UnlabelThreadPara*)arg;

	for(int a = 0; a < threadpara->unlabelData.size(); a++)
	{
		for(int i = 0; i < threadpara->v_domains.size(); i++)
		{
			threadpara->v_domains[i]->srcRM->rae1->buildTree(threadpara->unlabelData[a].first);
			threadpara->v_domains[i]->srcRM->rae2->buildTree(threadpara->unlabelData[a].second);
			threadpara->v_domains[i]->srcRM->softmax();
		}

		lbfgsfloatval_t src_ave = 0;

		for(int i = 0; i< threadpara->v_domains.size(); i++)
		{
			src_ave += threadpara->v_domains[i]->srcRM->outputLayer(0, 0);
		}

		src_ave /= threadpara->v_domains.size();

		threadpara->fx += (src_ave*2 - 1) * (src_ave*2 - 1);

		for(int i = 0; i < threadpara->v_domains.size(); i++)
		{
			threadpara->v_domains[i]->srcRM->trainOnUnlabel(src_ave, threadpara->v_domains.size());
		}
	}

	pthread_exit(NULL);
}

/*
static void* tgtUnlabelThread(void* arg)
{
	UnlabelThreadPara* threadpara = (UnlabelThreadPara*)arg;

	for(int a = 0; a < threadpara->unlabelData.size(); a++)
	{
		for(int i = 0; i < threadpara->v_domains.size(); i++)
		{
			threadpara->v_domains[i]->tgtRM->rae1->buildTree(threadpara->unlabelData[a].first);
			threadpara->v_domains[i]->tgtRM->rae2->buildTree(threadpara->unlabelData[a].second);
			threadpara->v_domains[i]->tgtRM->softmax();
		}

		lbfgsfloatval_t tgt_ave = 0;

		for(int i = 0; i< threadpara->v_domains.size(); i++)
		{
			tgt_ave += threadpara->v_domains[i]->tgtRM->outputLayer(0, 0);
		}

		tgt_ave /= threadpara->v_domains.size();

		threadpara->fx += (tgt_ave*2 - 1) * (tgt_ave*2 - 1);

		for(int i = 0; i < threadpara->v_domains.size(); i++)
		{
			threadpara->v_domains[i]->tgtRM->trainOnUnlabel(tgt_ave, threadpara->v_domains.size());
		}
	}

	pthread_exit(NULL);
}*/

//MixedDomain::MixedDomain(Parameter* para, vector<Domain*>& domains, RAE* srcRAE, RAE* tgtRAE)
MixedDomain::MixedDomain(Parameter* para, vector<Domain*>& domains, RAE* srcRAE)
{
	this->domains = domains;
	this->srcRAE = srcRAE;
	//this->tgtRAE = tgtRAE;
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
	double start, end;
	cout << "Start training......" << endl << endl;
	start = clock();

	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);
	param.max_iterations = atoi(para->getPara("IterationTime").c_str());
	int vecSize = atoi(para->getPara("WordVecSize").c_str());

	/*x = lbfgs_malloc((vecSize*2*2 + 2)*2*domains.size() + srcRAE->getRAEWeightSize()*2);
	Map<MatrixLBFGS>(x, (vecSize*2*2 + 2)*2*domains.size() + srcRAE->getRAEWeightSize()*2, 1).setRandom();*/

	x = lbfgs_malloc((vecSize*2*2 + 2)*domains.size() + srcRAE->getRAEWeightSize());
	Map<MatrixLBFGS>(x, (vecSize*2*2 + 2)*domains.size() + srcRAE->getRAEWeightSize(), 1).setRandom();

	lbfgsfloatval_t fx = 0;
	int ret = 0;
	//ret = lbfgs( (vecSize*2*2 + 2)*2*domains.size() + srcRAE->getRAEWeightSize()*2, x, &fx, evaluate, progress, this, &param);
	ret = lbfgs( (vecSize*2*2 + 2)*domains.size() + srcRAE->getRAEWeightSize(), x, &fx, evaluate, progress, this, &param);

	cout << "L-BFGS optimization terminated with status code = " << ret << endl;
	cout << " fx = " << fx << endl;

	//记录权重
	for(int i = 0; i < domains.size(); i++)
	{
		domains[i]->logWeights();
	}

	lbfgs_free(x);

	end = clock();
	cout << "The time of training is " << (end-start)/CLOCKS_PER_SEC << endl << endl;
}

lbfgsfloatval_t MixedDomain::_evaluate(const lbfgsfloatval_t* x,
									   lbfgsfloatval_t* g,
									   const int n,
									   const lbfgsfloatval_t step)
{
	lbfgsfloatval_t fx = 0;

	bool isDev = atoi(para->getPara("IsDev").c_str());
	bool isTrain = atoi(para->getPara("IsTrain").c_str());

	if(isUpdateRAE && isUpdateRM)
	{
		isUpdateRAE = true;
		isUpdateRM = false;
	}
	else
	{
		isUpdateRAE = !isUpdateRAE;
		isUpdateRM = !isUpdateRM;
	}

	srcRAE->updateWeights(x);
	//tgtRAE->updateWeights(x + srcRAE->getRAEWeightSize());

	for(int i = 0; i < amountOfDomains; i++)
	{
/*
		domains[i]->srcRM->rae = srcRAE->copy();
		domains[i]->tgtRM->rae = tgtRAE->copy();
		domains[i]->srcRM->updateWeights(x + srcRAE->getRAEWeightSize()*2 + i*2*domains[i]->srcRM->getRMWeightSize());
		domains[i]->tgtRM->updateWeights(x + srcRAE->getRAEWeightSize()*2 + (i*2+1)*domains[i]->srcRM->getRMWeightSize());
		wargs[i].g_RAE = g;
		wargs[i].g_RM = g + srcRAE->getRAEWeightSize()*2 + i*2*domains[i]->srcRM->getRMWeightSize();*/

		domains[i]->srcRM->rae = srcRAE->copy();
		domains[i]->srcRM->updateWeights(x + srcRAE->getRAEWeightSize() + i*domains[i]->srcRM->getRMWeightSize());
		wargs[i].g_RAE = g;
		wargs[i].g_RM = g + srcRAE->getRAEWeightSize() + i*domains[i]->srcRM->getRMWeightSize();
	}

	Start_Workers(train, wargs, amountOfDomains);

	int count = 0;
	for(int i = 0; i < amountOfDomains; i++)
	{
		count += wargs[i].domain->trainingData.size();
		fx += wargs[i].error;
	}

	fx /= count;

	//for(int i = 0; i < srcRAE->getRAEWeightSize()*2; i++)
	for(int i = 0; i < srcRAE->getRAEWeightSize(); i++)
	{
		g[i] /= count;
	}

	//Unlabel 训练
	lbfgsfloatval_t src_f = 0, tgt_f = 0;
	srcRAE->delToZero();
	//tgtRAE->delToZero();
	
	for(int i = 0; i < amountOfDomains; i++)
	{
		domains[i]->srcRM->delWeight.setZero();
		domains[i]->srcRM->delWeight_b.setZero();
	}
	

	if(isTrain)
	{
		getUnlabelData(para->getPara("SourceUnlabelTrainingData"));
	}
	else if(isDev)
	{
		getUnlabelData(para->getPara("SourceUnlabelDevData"));
	}

	int UnlabelThreadNum = atoi(para->getPara("UnlabelThreadNum").c_str());
	UnlabelThreadPara* threadpara = new UnlabelThreadPara[UnlabelThreadNum];
	int batchsize = unlabelData.size() / UnlabelThreadNum;

	if(batchsize == 0)
	{
		UnlabelThreadNum = 1;
	}

	for(int i = 0; i < UnlabelThreadNum; i++)
	{
		threadpara[i].fx = 0;
		for(int d = 0; d < amountOfDomains; d++)
		{
			threadpara[i].v_domains.push_back(domains[d]->copy());
		}

		if(i == UnlabelThreadNum-1)
		{
			threadpara[i].unlabelData.assign(unlabelData.begin()+i*batchsize, unlabelData.end());
			if(batchsize == 0)
			{
				threadpara[i].instance_num = unlabelData.size();
			}
			else
			{
				threadpara[i].instance_num = unlabelData.size()%batchsize;
			}
		}
		else
		{
			threadpara[i].unlabelData.assign(unlabelData.begin()+i*batchsize, unlabelData.begin()+(i+1)*batchsize);
			threadpara[i].instance_num = batchsize;
		}
	}

	pthread_t* pt = new pthread_t[UnlabelThreadNum];
	for (int a = 0; a < UnlabelThreadNum; a++) pthread_create(&pt[a], NULL, srcUnlabelThread, (void *)(threadpara + a));
	for (int a = 0; a < UnlabelThreadNum; a++) pthread_join(pt[a], NULL);

	for(int i = 0; i < UnlabelThreadNum; i++)
	{
		src_f += threadpara[i].fx;

		for(int d = 0; d < amountOfDomains; d++)
		{
			domains[d]->srcRM->delWeight += threadpara[i].v_domains[d]->srcRM->delWeight;
			domains[d]->srcRM->delWeight_b += threadpara[i].v_domains[d]->srcRM->delWeight_b;
			copyDelweights(domains[d]->srcRM->rae1, threadpara[i].v_domains[d]->srcRM->rae1);
			copyDelweights(domains[d]->srcRM->rae2, threadpara[i].v_domains[d]->srcRM->rae2);
		}
	}

	src_f /= unlabelData.size();
	for(int i = 0; i < amountOfDomains; i++)
	{
		domains[i]->srcRM->delWeight /= unlabelData.size();
		domains[i]->srcRM->delWeight_b /= unlabelData.size();
		copyDelweights(srcRAE, domains[i]->srcRM->rae1);
		copyDelweights(srcRAE, domains[i]->srcRM->rae2);
	}
	srcRAE->delWeight1 = srcRAE->delWeight1/(unlabelData.size() * amountOfDomains);
	srcRAE->delWeight1_b = srcRAE->delWeight1_b/(unlabelData.size() * amountOfDomains);

	/*if(isTrain)
	{
		getUnlabelData(para->getPara("TargetUnlabelTrainingData"));
	}
	else if(isDev)
	{
		getUnlabelData(para->getPara("TargetUnlabelDevData"));
	}


	UnlabelThreadNum = atoi(para->getPara("UnlabelThreadNum").c_str());
	batchsize = unlabelData.size() / UnlabelThreadNum;
	if(batchsize == 0)
	{
		UnlabelThreadNum = 1;
	}

	for(int i = 0; i < UnlabelThreadNum; i++)
	{
		threadpara[i].fx = 0;
		threadpara[i].unlabelData.clear();
		if(i == UnlabelThreadNum-1)
		{
			threadpara[i].unlabelData.assign(unlabelData.begin()+i*batchsize, unlabelData.end());
			if(batchsize == 0)
			{
				threadpara[i].instance_num = unlabelData.size();
			}
			else
			{
				threadpara[i].instance_num = unlabelData.size()%batchsize;
			}
		}
		else
		{
			threadpara[i].unlabelData.assign(unlabelData.begin()+i*batchsize, unlabelData.begin()+(i+1)*batchsize);
			threadpara[i].instance_num = batchsize;
		}
	}

	
	delete pt;
	pt = NULL;
	pt = new pthread_t[UnlabelThreadNum];
	for (int a = 0; a < UnlabelThreadNum; a++) pthread_create(&pt[a], NULL, tgtUnlabelThread, (void *)(threadpara + a));
	for (int a = 0; a < UnlabelThreadNum; a++) pthread_join(pt[a], NULL);

	for(int i = 0; i < UnlabelThreadNum; i++)
	{
		tgt_f += threadpara[i].fx;

		for(int d = 0; d < amountOfDomains; d++)
		{
			domains[d]->tgtRM->delWeight += threadpara[i].v_domains[d]->tgtRM->delWeight;
			domains[d]->tgtRM->delWeight_b += threadpara[i].v_domains[d]->tgtRM->delWeight_b;
			copyDelweights(domains[d]->tgtRM->rae1, threadpara[i].v_domains[d]->tgtRM->rae1);
			copyDelweights(domains[d]->tgtRM->rae2, threadpara[i].v_domains[d]->tgtRM->rae2);
		}
	}

	tgt_f /= unlabelData.size();
	for(int i = 0; i < amountOfDomains; i++)
	{
		domains[i]->tgtRM->delWeight /= unlabelData.size();
		domains[i]->tgtRM->delWeight_b /= unlabelData.size();
		copyDelweights(tgtRAE, domains[i]->tgtRM->rae1);
		copyDelweights(tgtRAE, domains[i]->tgtRM->rae2);
	}
	tgtRAE->delWeight1 = tgtRAE->delWeight1/(unlabelData.size() * amountOfDomains);
	tgtRAE->delWeight1_b = tgtRAE->delWeight1_b/(unlabelData.size() * amountOfDomains);

	for(int d = 0; d < amountOfDomains; d++)
	{
		int base = srcRAE->getRAEWeightSize()*2+d*domains[d]->getWeightsSize();
		domains[d]->update(g+base, g);
		srcRAE->delToZero();
		tgtRAE->delToZero();
	}

	fx -= (src_f+tgt_f);*/

	fx -= src_f;

	delete pt;
	pt = NULL;

	for(int i  = 0; i < UnlabelThreadNum; i++)
	{
		for(int d = 0; d < amountOfDomains; d++)
		{
			delete threadpara[i].v_domains[d];
		}
	}

	delete[] threadpara;
	threadpara = NULL;

	return fx;
}

vector<pair<int, map<string, string> > > MixedDomain::getTestData()
{
	string dataFile;
	bool isDev = atoi(para->getPara("IsDev").c_str());

	if(isDev)
	{
		dataFile = para->getPara("NewsDevFile");
	}
	else
	{
		dataFile = para->getPara("NewsTestFile");
	}

	ifstream in(dataFile.c_str(), ios::in);
	vector<pair<int, map<string, string> > > trainingData;

	string line;
	while(getline(in, line))
	{
		int order;
		map<string, string> m_tmp;
		vector<string> subOfLine = splitBySign(line);

		if(subOfLine[0] == "mono")
		{
			order = 1;
		}
		else if(subOfLine[0] == "invert")
		{
			order = 0;
		}

		for(int i = 1; i < subOfLine.size(); i++)
		{
			m_tmp.insert(make_pair(subOfLine[i].substr(0, 3), subOfLine[i].substr(4, subOfLine[i].size()-4)));
		}

		trainingData.push_back(make_pair(order, m_tmp));
	}

	in.close();
	return trainingData;
}

void MixedDomain::getUnlabelData(string filename)
{
	unlabelData.clear();
	ifstream in(filename.c_str(), ios::in);

	string line;
	while(getline(in, line))
	{
		int pos;
		pair<string, string> pss;

		if (line == "" || (pos = line.find("\t")) == string::npos)
		{
			continue;
		}

		pss.first = line.substr(0, pos);
		pos++;
		pss.second = line.substr(pos);

		unlabelData.push_back(pss);
	}
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
	double start, end;
	cout << "Starting testing..." << endl << endl;
	start = clock();

	Start_Workers(test, wargs, amountOfDomains);

	mixedTesting();

	end = clock();
	cout << "The time of testing is " << (end-start)/CLOCKS_PER_SEC << endl << endl;
}

void MixedDomain::mixedTesting()
{
	ofstream srcOut, tgtOut;
	vector<pair<int, map<string, string> > > trainingData;

	srcOut.open(string("./log/MixedDomain/srcNews.log").c_str(), ios::out);
	//tgtOut.open(string("./log/MixedDomain/tgtNews.log").c_str(), ios::out);

	trainingData = getTestData();

	bool isDev = atoi(para->getPara("IsDev").c_str());

	srcOut << "True value\t\tPredict value" << endl;
	//tgtOut << "True value\t\tPredict value" << endl;

	int srcCount = 0;
	int tgtCount = 0;

	//for(int i = trainingData.size()-100; i < trainingData.size(); i++)
	for(int i = 0; i < trainingData.size(); i++)
	{
		int srcMonoCount = 0, tgtMonoCount = 0;
		bool isSrcMono, isTgtMono;
		for(int d = 0; d < amountOfDomains; d++)
		{
			domains[d]->srcRM->getData(trainingData[i].second["ct1"], trainingData[i].second["ct2"]);
			//domains[d]->tgtRM->getData(trainingData[i].second["et1"], trainingData[i].second["et2"]);

			if(domains[d]->srcRM->outputLayer(0, 0) > domains[d]->srcRM->outputLayer(0, 1))
			{
				srcMonoCount++;
			}

/*
			if(domains[d]->tgtRM->outputLayer(0, 0) > domains[d]->tgtRM->outputLayer(0, 1))
			{
				tgtMonoCount++;
			}*/
		}
		if(srcMonoCount > amountOfDomains /2)
		{
			isSrcMono = true;
		}
		else
		{
			isSrcMono = false;
		}
		
/*
		if(tgtMonoCount > amountOfDomains /2)
		{
			isTgtMono = true;
		}
		else
		{
			isTgtMono = false;
		}*/

		if(isSrcMono)
		{
			if(trainingData[i].first == 1)
			{
				srcCount++;
			}
		}
		else
		{
			if(trainingData[i].first == 0)
			{
				srcCount++;
			}
		}

/*
		if(isTgtMono)
		{
			if(trainingData[i].first == 1)
			{
				tgtCount++;
			}
		}
		else
		{
			if(trainingData[i].first == 0)
			{
				tgtCount++;
			}
		}*/

		srcOut << trainingData[i].first << "\t\t";
		//tgtOut << trainingData[i].first << "\t\t";
		for(int d = 0; d < amountOfDomains; d++)
		{
			srcOut << "[" << domains[d]->srcRM->outputLayer(0,0) << " , " << domains[d]->srcRM->outputLayer(0,1) << "] ";
			//tgtOut << "[" << domains[d]->tgtRM->outputLayer(0,0) << " , " << domains[d]->tgtRM->outputLayer(0,1) << "] ";
		}
		srcOut << endl;
		//tgtOut << endl;
	}

	srcOut << "Precision: " << (lbfgsfloatval_t)srcCount/trainingData.size() << endl;
	//tgtOut << "Precision: " << (lbfgsfloatval_t)tgtCount/trainingData.size() << endl;

	srcOut.close();
	//tgtOut.close();
}

void train(worker_arg_t* arg)
{
	lbfgsfloatval_t start, end;

	Domain* d = arg->domain;

	d->loadTrainingData();

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

	//fx /= d->trainingData.size();

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

	start = clock();
	d->loadTestingData();
	end = clock();

	d->test();
}
