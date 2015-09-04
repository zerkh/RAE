#include "Domain.h"
#include "ThreadPara.h"

static void* DomainLBFGS::deepThread(void* arg)
{
	RMThreadPara* threadpara = (RMThreadPara*)arg;

	threadpara->lossVal = threadpara->d->_training(threadpara->g);

	pthread_exit(NULL);
}

Domain::Domain(Parameter* para, string domainName, RAE* srcRAE, RAE* tgtRAE)
{
	bool isDev = atoi(para->getPara("IsDev").c_str());
	bool isTrain = atoi(para->getPara("IsTrain").c_str());
	bool isTest = atoi(para->getPara("IsTest").c_str());

	this->para = para;

	if(isDev)
	{
		dataFile = para->getPara(domainName + "DevDataFile");
	}
	else if(isTrain)
	{
		dataFile = para->getPara(domainName + "DataFile");
	}
	else if(isTest)
	{
		dataFile = para->getPara(domainName + "TestDataFile");
	}

	iterTime = atoi(para->getPara("IterationTime").c_str());

	this->domainName = domainName;
	out.open(string("./log/"+ domainName + "/" + domainName+".log").c_str(), ios::out);

	srcRM = new ReorderModel(para, srcRAE);
	tgtRM = new ReorderModel(para, tgtRAE);
}

int Domain::getWeightsSize()
{
	return (srcRM->getRMWeightSize()*2);
}

void Domain::upData(lbfgsfloatval_t* g)
{
	Map<MatrixLBFGS> g_srcWeights(g, srcRM->weights.rows(), srcRM->weights.cols());
	Map<MatrixLBFGS> g_srcWeights_b(g+srcRM->weights.rows()*srcRM->weights.cols(), srcRM->weights_b.rows(), srcRM->weights_b.cols());

	int base = srcRM->getRMWeightSize();
	Map<MatrixLBFGS> g_tgtWeights(g+base, srcRM->weights.rows(), srcRM->weights.cols());
	Map<MatrixLBFGS> g_tgtWeights_b(g+base+srcRM->weights.rows()*srcRM->weights.cols(), srcRM->weights_b.rows(), srcRM->weights_b.cols());

	g_srcWeights += srcRM->delWeight;
	g_srcWeights_b += srcRM->delWeight_b;

	g_tgtWeights += tgtRM->delWeight;
	g_tgtWeights_b += tgtRM->delWeight_b;

	srcRM->delWeight.setZero();
	srcRM->delWeight_b.setZero();
	tgtRM->delWeight.setZero();
	tgtRM->delWeight_b.setZero();

	delete srcRM->rae1;
	delete srcRM->rae2;
	delete tgtRM->rae1;
	delete tgtRM->rae2;

	srcRM->rae1 = srcRM->rae->copy();
	srcRM->rae2 = srcRM->rae->copy();

	tgtRM->rae1 = tgtRM->rae->copy();
	tgtRM->rae2 = tgtRM->rae->copy();
}

//读取单领域训练数据
void Domain::loadTrainingData()
{
	ifstream in(dataFile.c_str(), ios::in);

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
}

Domain* Domain::copy()
{
	Domain* d = new Domain(para, domainName, srcRM->rae, tgtRM->rae);

	d->srcRM = srcRM->copy();
	d->tgtRM = tgtRM->copy();

	return d;
}

lbfgsfloatval_t Domain::_evaluate(const lbfgsfloatval_t* x,
							lbfgsfloatval_t* g,
							const int n,
							const lbfgsfloatval_t step)
{
	lbfgsfloatval_t fx = 0;

	srcRM->updateWeights(x, 0);
	tgtRM->updateWeights(x, srcRM->getRMWeightSize());

	int RMThreadNum = atoi(para->getPara("RMThreadNum").c_str());
	RMThreadPara* threadpara = new RMThreadPara[RMThreadNum];
	int batchsize = trainingData.size() / RMThreadNum;

	for(int i = 0; i < RMThreadNum; i++)
	{
		threadpara[i].d = this->copy();
		threadpara[i].g = lbfgs_malloc(getWeightsSize());
		if(i == RMThreadNum-1)
		{
			threadpara[i].d->trainingData.assign(trainingData.begin()+i*batchsize, trainingData.end());
			threadpara[i].instance_num = trainingData.size()%batchsize;
		}
		else
		{
			threadpara[i].d->trainingData.assign(trainingData.begin()+i*batchsize, trainingData.begin()+(i+1)*batchsize);
			threadpara[i].instance_num = batchsize;
		}
	}
	pthread_t* pt = new pthread_t[RMThreadNum];
	for (int a = 0; a < RMThreadNum; a++) pthread_create(&pt[a], NULL, DomainLBFGS::deepThread, (void *)(threadpara + a));
	for (int a = 0; a < RMThreadNum; a++) pthread_join(pt[a], NULL);

	for(int i = 0; i < RMThreadNum; i++)
	{
		fx += threadpara[i].lossVal;
		for(int elem = 0; elem < getWeightsSize(); elem++)
		{
			g[elem] += threadpara[i].g[elem];
		}
	}

	fx /= trainingData.size();
	for(int elem = 0; elem < getWeightsSize(); elem++)
	{
		g[elem] /= trainingData.size();
	}

	delete pt;
	pt = NULL;

	for(int i  = 0; i < RMThreadNum; i++)
	{
		lbfgs_free(threadpara[i].g);
		delete threadpara[i].d;
	}
	delete threadpara;
	threadpara = NULL;

	return fx;
}

int Domain::_progress(const lbfgsfloatval_t *x,
					const lbfgsfloatval_t *g,
					const lbfgsfloatval_t fx,
					const lbfgsfloatval_t xnorm,
					const lbfgsfloatval_t gnorm,
					const lbfgsfloatval_t step,
					int n,
					int k,
					int ls)
{
	out << "Iteration: " << k << endl;
	out << "Loss Value: " << fx << endl;

	return 0;
}

lbfgsfloatval_t Domain::_training(lbfgsfloatval_t* g)
{
	lbfgsfloatval_t error = 0;

	for(int i = 0; i < trainingData.size(); i++)
	{
		error += loss(i);

		//获取实例
		srcRM->getData(trainingData[i].second["ct1"], trainingData[i].second["ct2"]);
		tgtRM->getData(trainingData[i].second["et1"], trainingData[i].second["et2"]);

		//对调序模型求导(Edis)
		srcRM->trainRM(tgtRM->outputLayer, true);
		tgtRM->trainRM(srcRM->outputLayer, true);

		//Ereo
		MatrixLBFGS mono = MatrixLBFGS(1,2);
		MatrixLBFGS invert = MatrixLBFGS(1,2);

		for (int j = 0; j < 2; j++)
		{
			mono(0, j) = 1-j;
			invert(0, j) = j;
		}

		if(trainingData[i].first == 1)
		{
			srcRM->trainRM(mono, false);
			tgtRM->trainRM(mono, false);
		}
		else
		{
			srcRM->trainRM(invert, false);
			tgtRM->trainRM(invert, false);
		}

		upData(g);
	}

	return error/trainingData.size();
}

void Domain::training()
{
	lbfgs_parameter_t param;
	lbfgs_parameter_init(&param);
	param.max_iterations = iterTime;
	x = lbfgs_malloc(srcRM->getRMWeightSize() + tgtRM->getRMWeightSize());
	Map<MatrixLBFGS>(x, srcRM->getRMWeightSize() + tgtRM->getRMWeightSize(), 1).setRandom();

	lbfgsfloatval_t fx = 0;
	int ret = 0;
	ret = lbfgs(getWeightsSize(), x, &fx, DomainLBFGS::evaluate, DomainLBFGS::progress, this, &param);

	cout << "L-BFGS optimization terminated with status code = " << ret << endl;
	cout << " fx = " << fx << endl;

	//记录权重
	logWeights();
	lbfgs_free(x);
}

//获取单领域的loss value
lbfgsfloatval_t Domain::loss(int ind)
{
	lbfgsfloatval_t lossVal = 0;

	srcRM->getData(trainingData[ind].second["ct1"], trainingData[ind].second["ct2"]);
	tgtRM->getData(trainingData[ind].second["et1"], trainingData[ind].second["et2"]);

	lossVal += GAMMA * (srcRM->outputLayer(0, 0)*tgtRM->outputLayer(0, 0) + srcRM->outputLayer(0, 1)*tgtRM->outputLayer(0, 1));

	//cout << "Src softmax: [" << srcRM->softmaxLayer(0, 0) << " , " << srcRM->softmaxLayer(0, 1) << "]" << endl;	
	//cout << "Tgt softmax: [" << tgtRM->softmaxLayer(0, 0) << " , " << tgtRM->softmaxLayer(0, 1) << "]" << endl; 	
	//cout << "Src output: [" << srcRM->outputLayer(0, 0) << " , " << srcRM->outputLayer(0, 1) << "]" << endl; 
	//cout << "Tgt output: [" << tgtRM->outputLayer(0, 0) << " , " << tgtRM->outputLayer(0, 1) << "]" << endl; 

	if(trainingData[ind].first == 1)
	{
		lossVal += BETA * srcRM->softmaxLayer(0, 0) * -1.0;
		lossVal += BETA * tgtRM->softmaxLayer(0, 0) * -1.0;
	}
	else
	{
		lossVal += BETA * srcRM->softmaxLayer(0, 1) * -1.0;
		lossVal += BETA * tgtRM->softmaxLayer(0, 1) * -1.0;
	}

	//cout << "After Ereo loss: " << lossVal << endl;	

	lossVal += ZETA * (srcRM->decay() + tgtRM->decay());

	//cout << "After Decay loss: " << lossVal << endl;

	return lossVal;
}

void Domain::loadWeights()
{
	ifstream src(string("./log/"+ domainName + "/" + "src"+domainName+"Weights.log").c_str(), ios::in);
	ifstream tgt(string("./log/"+ domainName + "/" + "tgt"+domainName+"Weights.log").c_str(), ios::in);

	bool rae_w1 = false;
	bool rae_b1 = false;
	bool rae_w2 = false;
	bool rae_b2 = false;
	bool rm_w = false;
	bool rm_b = false;
	int row = 0;

	string line;
	while(getline(src, line))
	{
		if(line.find("W:") == 0)
		{
			row = 0;
			rm_w = true;
			continue;
		}
		if(line.find("b:") == 0)
		{
			row = 0;
			rm_w = false;
			rm_b = true;
			continue;
		}
		if(line.find("W1:") == 0)
		{
			row = 0;
			rm_b = false;
			rae_w1 = true;
			continue;
		}
		if(line.find("W2:") == 0)
		{
			row = 0;
			rae_b1 = false;
			rae_w2 = true;
			continue;
		}
		if(line.find("b1:") == 0)
		{
			row = 0;
			rae_w1 = false;
			rae_b1 = true;
			continue;
		}
		if(line.find("b2:") == 0)
		{
			row = 0;
			rae_w2 = false;
			rae_b2 = true;
			continue;
		}

		if(rm_w)
		{
			stringstream ss(line);

			for(int col = 0; col < srcRM->weights.cols(); col++)
			{
				ss >> srcRM->weights(row, col);
			}
		}
		if(rm_b)
		{
			stringstream ss(line);

			for(int col = 0; col < srcRM->weights_b.cols(); col++)
			{
				ss >> srcRM->weights_b(row, col);
			}
		}
		if(rae_w1)
		{
			stringstream ss(line);

			for(int col = 0; col < srcRM->rae->weights1.cols(); col++)
			{
				ss >> srcRM->rae->weights1(row, col);
			}
		}
		if(rae_w2)
		{
			stringstream ss(line);

			for(int col = 0; col < srcRM->rae->weights2.cols(); col++)
			{
				ss >> srcRM->rae->weights2(row, col);
			}
		}
		if(rae_b2)
		{
			stringstream ss(line);

			for(int col = 0; col < srcRM->rae->weights_b2.cols(); col++)
			{
				ss >> srcRM->rae->weights_b2(row, col);
			}
		}
		if(rae_b1)
		{
			stringstream ss(line);

			for(int col = 0; col < srcRM->rae->weights_b1.cols(); col++)
			{
				ss >> srcRM->rae->weights_b1(row, col);
			}
		}

		row++;
	}
	srcRM->rae1 = srcRM->rae->copy();
	srcRM->rae2 = srcRM->rae->copy();
	src.close();

	rae_b2 = false;
	while(getline(tgt, line))
	{
		if(line.find("W:") == 0)
		{
			row = 0;
			rm_w = true;
			continue;
		}
		if(line.find("b:") == 0)
		{
			row = 0;
			rm_w = false;
			rm_b = true;
			continue;
		}
		if(line.find("W1:") == 0)
		{
			row = 0;
			rm_b = false;
			rae_w1 = true;
			continue;
		}
		if(line.find("W2:") == 0)
		{
			row = 0;
			rae_b1 = false;
			rae_w2 = true;
			continue;
		}
		if(line.find("b1:") == 0)
		{
			row = 0;
			rae_w1 = false;
			rae_b1 = true;
			continue;
		}
		if(line.find("b2:") == 0)
		{
			row = 0;
			rae_w2 = false;
			rae_b2 = true;
			continue;
		}

		if(rm_w)
		{
			stringstream ss(line);

			for(int col = 0; col < tgtRM->weights.cols(); col++)
			{
				ss >> tgtRM->weights(row, col);
			}
		}
		if(rm_b)
		{
			stringstream ss(line);

			for(int col = 0; col < tgtRM->weights_b.cols(); col++)
			{
				ss >> tgtRM->weights_b(row, col);
			}
		}
		if(rae_w1)
		{
			stringstream ss(line);

			for(int col = 0; col < tgtRM->rae->weights1.cols(); col++)
			{
				ss >> tgtRM->rae->weights1(row, col);
			}
		}
		if(rae_w2)
		{
			stringstream ss(line);

			for(int col = 0; col < tgtRM->rae->weights2.cols(); col++)
			{
				ss >> tgtRM->rae->weights2(row, col);
			}
		}
		if(rae_b2)
		{
			stringstream ss(line);

			for(int col = 0; col < tgtRM->rae->weights_b2.cols(); col++)
			{
				ss >> tgtRM->rae->weights_b2(row, col);
			}
		}
		if(rae_b1)
		{
			stringstream ss(line);

			for(int col = 0; col < tgtRM->rae->weights_b1.cols(); col++)
			{
				ss >> tgtRM->rae->weights_b1(row, col);
			}
		}

		row++;
	}
	tgtRM->rae1 = tgtRM->rae->copy();
	tgtRM->rae2 = tgtRM->rae->copy();
	tgt.close();
}

void Domain::loadTestingData()
{
	ifstream in(dataFile.c_str(), ios::in);

	trainingData.clear();
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

	loadWeights();
}

void Domain::logWeights()
{	
	srcWLog.open(string("./log/"+ domainName + "/" + "src"+domainName+"Weights.log").c_str(), ios::out);
	tgtWLog.open(string("./log/"+ domainName + "/" + "tgt"+domainName+"Weights.log").c_str(), ios::out);

	srcWLog << "RM: \nW: \n";
	tgtWLog << "RM: \nW: \n";
	srcWLog << srcRM->weights << endl;
	tgtWLog << tgtRM->weights << endl;

	srcWLog << "b: \n";
	tgtWLog << "b: \n";
	srcWLog << srcRM->weights_b << endl;
	tgtWLog << tgtRM->weights_b << endl;

	srcWLog << "RAE: \nW1: \n";
	tgtWLog << "RAE: \nW1: \n";
	srcWLog << srcRM->rae->weights1 << endl;
	tgtWLog << tgtRM->rae->weights1 << endl;

	srcWLog << "b1: \n";
	tgtWLog << "b1: \n";
	srcWLog << srcRM->rae->weights_b1 << endl;
	tgtWLog << tgtRM->rae->weights_b1 << endl;

	srcWLog << "W2: \n";
	tgtWLog << "W2: \n";
	srcWLog << srcRM->rae->weights2 << endl;
	tgtWLog << tgtRM->rae->weights2 << endl;

	srcWLog << "b2: \n";
	tgtWLog << "b2: \n";
	srcWLog << srcRM->rae->weights_b2 << endl;
	tgtWLog << tgtRM->rae->weights_b2 << endl;

	srcWLog.close();
	srcWLog.close();
}

void Domain::test()
{
	srcOut.open(string("./log/"+ domainName + "/" + "src"+domainName+".log").c_str(), ios::out);
	tgtOut.open(string("./log/"+ domainName + "/" + "tgt"+domainName+".log").c_str(), ios::out);

	srcOut << "True value\t\tPredict value" << endl;
	tgtOut << "True value\t\tPredict value" << endl;

	int srcCount = 0;
	int tgtCount = 0;

	//for(int i = trainingData.size()-100; i < trainingData.size(); i++)
	for(int i = 0; i < trainingData.size(); i++)
	{
		srcRM->getData(trainingData[i].second["ct1"], trainingData[i].second["ct2"]);
		tgtRM->getData(trainingData[i].second["et1"], trainingData[i].second["et2"]);

		if(srcRM->outputLayer(0, 0) > srcRM->outputLayer(0,1))
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

		if(tgtRM->outputLayer(0, 0) > tgtRM->outputLayer(0,1))
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
		}

		srcOut << trainingData[i].first << "\t\t[" << srcRM->outputLayer(0,0) << " , " << srcRM->outputLayer(0,1) << "]" << endl;
		tgtOut << trainingData[i].first << "\t\t[" << tgtRM->outputLayer(0,0) << " , " << tgtRM->outputLayer(0,1) << "]" << endl;
	}

	srcOut << "Precision: " << (lbfgsfloatval_t)srcCount/trainingData.size() << endl;
	tgtOut << "Precision: " << (lbfgsfloatval_t)tgtCount/trainingData.size() << endl;

	srcOut.close();
	tgtOut.close();
}
