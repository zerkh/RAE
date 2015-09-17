#include "Domain.h"

Domain::Domain(Parameter* para, string domainName, RAE* srcRAE)
{
	bool isDev = atoi(para->getPara("IsDev").c_str());
	bool isTrain = atoi(para->getPara("IsTrain").c_str());
	bool isTest = atoi(para->getPara("IsTest").c_str());

	this->srcRAE = srcRAE;

	this->para = para;

	if(isDev)
	{
		dataFile = para->getPara(domainName + "DevTrainFile");
	}
	else if(isTrain)
	{
		dataFile = para->getPara(domainName + "DataFile");
	}
	else if(isTest)
	{
		dataFile = para->getPara(domainName + "TestDataFile");
	}

	this->domainName = domainName;
	out.open(string("./log/"+ domainName + "/" + domainName+".log").c_str(), ios::out);

	srcRM = new ReorderModel(para, srcRAE);
}

Domain* Domain::copy()
{
	Domain* d = new Domain(para, domainName, srcRM->rae);

	d->srcRM = srcRM->copy();

	return d;
}

int Domain::getWeightsSize()
{
	return (srcRM->getRMWeightSize());
}

void Domain::update(lbfgsfloatval_t* g_RM, lbfgsfloatval_t* g_RAE)
{
	Map<MatrixLBFGS> g_srcWeights(g_RM, srcRM->weights.rows(), srcRM->weights.cols());
	Map<MatrixLBFGS> g_srcWeights_b(g_RM + srcRM->weights.rows()*srcRM->weights.cols(), srcRM->weights_b.rows(), srcRM->weights_b.cols());
	Map<MatrixLBFGS> g_srcRAEWeights1(g_RAE, srcRM->rae->weights1.rows(), srcRM->rae->weights1.cols());
	Map<MatrixLBFGS> g_srcRAEWeights_b1(g_RAE + srcRAE->weights1.rows()*srcRAE->weights1.cols(), srcRAE->weights_b1.rows(), srcRAE->weights_b1.cols());
	Map<MatrixLBFGS> g_srcRAEWeights2(g_RAE + srcRAE->weights1.rows()*srcRAE->weights1.cols()+
		srcRAE->weights_b1.rows()*srcRAE->weights_b1.cols(),
		srcRAE->weights2.rows(), srcRAE->weights2.cols());
	Map<MatrixLBFGS> g_srcRAEWeights_b2(g_RAE + srcRAE->weights1.rows()*srcRAE->weights1.cols()+
		srcRAE->weights_b1.rows()*srcRAE->weights_b1.cols()+
		srcRAE->weights2.rows()*srcRAE->weights2.cols(),
		srcRAE->weights_b2.rows(), srcRAE->weights_b2.cols());

	if(isUpdateRM)
	{
		g_srcWeights += srcRM->delWeight;
		g_srcWeights_b += srcRM->delWeight_b;
	}

	if(isUpdateRAE)
	{
		g_srcRAEWeights1 += srcRAE->delWeight1;
		g_srcRAEWeights_b1 += srcRAE->delWeight1_b;
		g_srcRAEWeights2 += srcRAE->delWeight2;
		g_srcRAEWeights_b2 += srcRAE->delWeight2_b;
	}

	srcRM->delWeight.setZero();
	srcRM->delWeight_b.setZero();

	delete srcRM->rae1;
	delete srcRM->rae2;

	srcRM->rae1 = srcRM->rae->copy();
	srcRM->rae2 = srcRM->rae->copy();
}

//读取单领域训练数据
void Domain::loadTrainingData()
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
}

lbfgsfloatval_t Domain::training(lbfgsfloatval_t* g_RM, lbfgsfloatval_t* g_RAE)
{
	lbfgsfloatval_t error = 0;

	for(int i = 0; i < trainingData.size(); i++)
	{
		error += loss(i);

		//获取实例
		srcRM->getData(trainingData[i].second["ct1"], trainingData[i].second["ct2"]);

		//对rae求导
		srcRM->rae1->trainRecError();
		srcRM->rae2->trainRecError();

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
			srcRM->trainRM(mono, EREO);
		}
		else
		{
			srcRM->trainRM(invert, EREO);
		}

		copyDelweights(srcRAE, srcRM->rae1);
		copyDelweights(srcRAE, srcRM->rae2);

		update(g_RM, g_RAE);
	}

	return error;
}

//获取单领域的loss value
lbfgsfloatval_t Domain::loss(int ind)
{
	lbfgsfloatval_t lossVal = 0;

	srcRM->getData(trainingData[ind].second["ct1"], trainingData[ind].second["ct2"]);

	lossVal += ALPHA * srcRM->rae1->loss();
	lossVal += ALPHA * srcRM->rae2->loss();


	if(trainingData[ind].first == 1)
	{
		lossVal += BETA * srcRM->softmaxLayer(0, 0) * -1.0;
	}
	else
	{
		lossVal += BETA * srcRM->softmaxLayer(0, 1) * -1.0;
	}

	lossVal += ZETA * (srcRM->decay());

	return lossVal;
}

void Domain::loadWeights()
{
	ifstream src(string("./log/"+ domainName + "/" + "src"+domainName+"Weights.log").c_str(), ios::in);

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
		if(line.find("RAE") == 0 || line.find("RM") == 0)
		{
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
}

void Domain::loadTestingData()
{
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

	srcWLog << "RM: \nW: \n";
	srcWLog << srcRM->weights << endl;

	srcWLog << "b: \n";
	srcWLog << srcRM->weights_b << endl;

	srcWLog << "RAE: \nW1: \n";
	srcWLog << srcRM->rae->weights1 << endl;

	srcWLog << "b1: \n";
	srcWLog << srcRM->rae->weights_b1 << endl;

	srcWLog << "W2: \n";
	srcWLog << srcRM->rae->weights2 << endl;

	srcWLog << "b2: \n";
	srcWLog << srcRM->rae->weights_b2 << endl;

	srcWLog.close();
}

void Domain::test()
{
	srcOut.open(string("./log/"+ domainName + "/" + "src"+domainName+".log").c_str(), ios::out);

	ofstream mixOut;
	bool isDev = atoi(para->getPara("IsDev").c_str());

	srcOut << "True value\t\tPredict value" << endl;

	int srcCount = 0;
	int tgtCount = 0;

	for(int i = 0; i < trainingData.size(); i++)
	{
		srcRM->getData(trainingData[i].second["ct1"], trainingData[i].second["ct2"]);

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

		srcOut << trainingData[i].first << "\t\t[" << srcRM->outputLayer(0,0) << " , " << srcRM->outputLayer(0,1) << "]" << endl;
	}

	srcOut << "Precision: " << (lbfgsfloatval_t)srcCount/trainingData.size() << endl;

	srcOut.close();
}
