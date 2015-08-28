#include "Domain.h"

Domain::Domain(Parameter* para, string domainName, WordVec* srcWords, WordVec* tgtWords)
{
	bool isDev = atoi(para->getPara("IsDev").c_str());
	bool isTrain = atoi(para->getPara("IsTrain").c_str());
	bool isTest = atoi(para->getPara("IsTest").c_str());

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

	srcRM = new ReorderModel(para, srcWords);
	tgtRM = new ReorderModel(para, tgtWords);

	x = lbfgs_malloc(srcRM->getRMWeightSize() + tgtRM->getRMWeightSize() + srcRM->rae->getRAEWeightSize() + tgtRM->rae->getRAEWeightSize());
	Map<MatrixLBFGS>(x, srcRM->getRMWeightSize() + tgtRM->getRMWeightSize() + srcRM->rae->getRAEWeightSize() + tgtRM->rae->getRAEWeightSize(), 1).setRandom();
}

int Domain::getWeightsSize()
{
	return (srcRM->getRMWeightSize()*2+srcRM->rae->getRAEWeightSize()*2);
}

void Domain::upData(lbfgsfloatval_t* g)
{
	Map<MatrixLBFGS> g_srcWeights(g, srcRM->weights.rows(), srcRM->weights.cols());
	Map<MatrixLBFGS> g_srcWeights_b(g+srcRM->weights.rows()*srcRM->weights.cols(), srcRM->weights_b.rows(), srcRM->weights_b.cols());
	Map<MatrixLBFGS> g_srcRAEWeights1(g+srcRM->getRMWeightSize(), srcRM->rae->weights1.rows(), srcRM->rae->weights1.cols());
	Map<MatrixLBFGS> g_srcRAEWeights_b1(g+srcRM->getRMWeightSize()+srcRM->rae->weights1.rows()*srcRM->rae->weights1.cols(), srcRM->rae->weights_b1.rows(), srcRM->rae->weights_b1.cols());
	Map<MatrixLBFGS> g_srcRAEWeights2(g+srcRM->getRMWeightSize()+srcRM->rae->weights1.rows()*srcRM->rae->weights1.cols()+
									srcRM->rae->weights_b1.rows()*srcRM->rae->weights_b1.cols(),
									srcRM->rae->weights2.rows(), srcRM->rae->weights2.cols());
	Map<MatrixLBFGS> g_srcRAEWeights_b2(g+srcRM->getRMWeightSize()+srcRM->rae->weights1.rows()*srcRM->rae->weights1.cols()+
									srcRM->rae->weights_b1.rows()*srcRM->rae->weights_b1.cols()+
									srcRM->rae->weights2.rows()*srcRM->rae->weights2.cols(),
									srcRM->rae->weights_b2.rows(), srcRM->rae->weights_b2.cols());

	int base = srcRM->getRMWeightSize() + srcRM->rae->getRAEWeightSize();
	Map<MatrixLBFGS> g_tgtWeights(g+base, srcRM->weights.rows(), srcRM->weights.cols());
	Map<MatrixLBFGS> g_tgtWeights_b(g+base+srcRM->weights.rows()*srcRM->weights.cols(), srcRM->weights_b.rows(), srcRM->weights_b.cols());
	Map<MatrixLBFGS> g_tgtRAEWeights1(g+base+srcRM->getRMWeightSize(), srcRM->rae->weights1.rows(), srcRM->rae->weights1.cols());
	Map<MatrixLBFGS> g_tgtRAEWeights_b1(g+base+srcRM->getRMWeightSize()+srcRM->rae->weights1.rows()*srcRM->rae->weights1.cols(), srcRM->rae->weights_b1.rows(), srcRM->rae->weights_b1.cols());
	Map<MatrixLBFGS> g_tgtRAEWeights2(g+base+srcRM->getRMWeightSize()+srcRM->rae->weights1.rows()*srcRM->rae->weights1.cols()+
		srcRM->rae->weights_b1.rows()*srcRM->rae->weights_b1.cols(),
		srcRM->rae->weights2.rows(), srcRM->rae->weights2.cols());
	Map<MatrixLBFGS> g_tgtRAEWeights_b2(g+base+srcRM->getRMWeightSize()+srcRM->rae->weights1.rows()*srcRM->rae->weights1.cols()+
		srcRM->rae->weights_b1.rows()*srcRM->rae->weights_b1.cols()+
		srcRM->rae->weights2.rows()*srcRM->rae->weights2.cols(),
		srcRM->rae->weights_b2.rows(), srcRM->rae->weights_b2.cols());

	g_srcWeights += srcRM->delWeight;
	g_srcWeights_b += srcRM->delWeight_b;
	g_srcRAEWeights1 += srcRM->rae->delWeight1;
	g_srcRAEWeights_b1 += srcRM->rae->delWeight1_b;
	g_srcRAEWeights2 += srcRM->rae->delWeight2;
	g_srcRAEWeights_b2 += srcRM->rae->delWeight2_b;

	g_tgtWeights += tgtRM->delWeight;
	g_tgtWeights_b += tgtRM->delWeight_b;
	g_tgtRAEWeights1 += tgtRM->rae->delWeight1;
	g_tgtRAEWeights_b1 += tgtRM->rae->delWeight1_b;
	g_tgtRAEWeights2 += tgtRM->rae->delWeight2;
	g_tgtRAEWeights_b2 += tgtRM->rae->delWeight2_b;

	srcRM->delWeight.setZero();
	srcRM->delWeight_b.setZero();
	tgtRM->delWeight.setZero();
	tgtRM->delWeight_b.setZero();

	srcRM->rae1 = srcRM->rae;
	srcRM->rae2 = srcRM->rae;

	tgtRM->rae1 = tgtRM->rae;
	tgtRM->rae2 = tgtRM->rae;
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

lbfgsfloatval_t Domain::_evaluate(const lbfgsfloatval_t* x,
							lbfgsfloatval_t* g,
							const int n,
							const lbfgsfloatval_t step)
{
	lbfgsfloatval_t fx = 0;

	srcRM->updateWeights(x, 0);
	tgtRM->updateWeights(x, srcRM->getRMWeightSize()+srcRM->rae->getRAEWeightSize());

	fx += _training(g);

	for(int i = 0; i < getWeightsSize(); i++)
	{
		g[i] /= trainingData.size();
	}

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
	cout << "Iteration: " << k << endl;
	cout << "Loss Value: " << fx << endl;

	return 0;
}

lbfgsfloatval_t Domain::_training(lbfgsfloatval_t* g)
{
	lbfgsfloatval_t error = 0;

	for(int i = 0; i < trainingData.size(); i++)
	{
		error += loss(i);

		//cout << "Domain getData" << endl;
		//获取实例
		srcRM->getData(trainingData[i].second["ct1"], trainingData[i].second["ct2"]);
		tgtRM->getData(trainingData[i].second["et1"], trainingData[i].second["et2"]);
		//cout << "Domain getData" << endl;			

		//cout << "domain reconstruct error" << endl;
		//对rae求导
		srcRM->rae1->trainRecError();
		srcRM->rae2->trainRecError();
		tgtRM->rae1->trainRecError();
		tgtRM->rae2->trainRecError();
		//cout << "domain reconstruct error" << endl;

		//cout << "Domain Edis" << endl;
		//对调序模型求导(Edis)
		srcRM->trainRM(tgtRM->outputLayer, true);
		tgtRM->trainRM(srcRM->outputLayer, true);
		//cout << "Domain Edis" << endl;

		//cout << "Domain Rreo" << endl;
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

	lbfgsfloatval_t fx = 0;
	int ret = 0;
	ret = lbfgs(getWeightsSize(), x, &fx, evaluate, progress, this, &param);

	cout << "L-BFGS optimization terminated with status code = " << ret << endl;
	cout << " fx = " << fx << endl;

	//记录权重
	logWeights();
}

//获取单领域的loss value
lbfgsfloatval_t Domain::loss(int ind)
{
	lbfgsfloatval_t lossVal = 0;

	srcRM->getData(trainingData[ind].second["ct1"], trainingData[ind].second["ct2"]);
	tgtRM->getData(trainingData[ind].second["et1"], trainingData[ind].second["et2"]);

	lossVal += ALPHA * srcRM->rae1->loss();
	lossVal += ALPHA * srcRM->rae2->loss();
	lossVal += ALPHA * tgtRM->rae1->loss();
	lossVal += ALPHA * tgtRM->rae2->loss();

	for(int i = 0; i < 2; i++)
	{
		lossVal += GAMMA * pow(srcRM->outputLayer(0, i) - tgtRM->outputLayer(0, i), 2) / 2;
	}

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

	lossVal += ZETA * (srcRM->decay() + tgtRM->decay() + srcRM->rae->decay() + tgtRM->rae->decay());

	//cout << "After Decay loss: " << lossVal << endl;

	return lossVal;
}

void Domain::loadWeights()
{
	ifstream src(string("src"+domainName+"Weights.log").c_str(), ios::in);
	ifstream tgt(string("tgt"+domainName+"Weights.log").c_str(), ios::in);

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
