#include "RAE.h"
#include "ThreadPara.h"

static void* RAELBFGS::deepThread(void* arg)
{
	RAEThreadPara* threadpara = (RAEThreadPara*)arg;

	threadpara->lossVal = threadpara->cRAE->_training(threadpara->g);

	pthread_exit(NULL);
}

RAE::RAE(Parameter* para, WordVec* words, int RAEType)
{
	vecSize = atoi(para->getPara("WordVecSize").c_str());
	this->words = words;

	RAETree = NULL;
	x = NULL;

	this->RAEType = RAEType;
	this->para = para;

	delWeight1 = MatrixLBFGS(vecSize, 2*vecSize);
	delWeight1_b = MatrixLBFGS(1, vecSize);
	delWeight2 = MatrixLBFGS(2*vecSize, vecSize);
	delWeight2_b = MatrixLBFGS(1, 2*vecSize);

	delWeight1.setZero();
	delWeight1_b.setZero();
	delWeight2.setZero();
	delWeight2_b.setZero();
}

int RAE::getRAEWeightSize()
{
	return (vecSize*vecSize*2*2 + vecSize + vecSize*2);
}

RAE::RAE(int size)
{
	RAETree = NULL;
	this->vecSize = size;
	delWeight1 = MatrixLBFGS(vecSize, vecSize*2);
	delWeight1_b = MatrixLBFGS(1, vecSize);
	delWeight2 = MatrixLBFGS(vecSize*2, vecSize);
	delWeight2_b = MatrixLBFGS(1, vecSize*2);

	delWeight1.setZero();
	delWeight1_b.setZero();
	delWeight2.setZero();
	delWeight2_b.setZero();
}

lbfgsfloatval_t RAE::loss()
{
	Node* tmpNode = this->RAETree->getRoot();
	lbfgsfloatval_t val = 0;

	while(tmpNode->getNodeType() != BASED_NODE)
	{
		val += tmpNode->getRecError();

		tmpNode = tmpNode->getLeftChildNode();
	}

	return val + decay();
}

//显示参数
void RAE::showWeights()
{
	cout << "W1\t" << "Row: " << weights1.rows() << "\tCol: " << weights1.cols() << endl;
	cout << weights1 << endl;

	cout << "B1\t" << "Row: " << weights_b1.rows() << "\tCol: " << weights_b1.cols() << endl;
	cout << weights_b1 << endl;

	cout << "W2\t" << "Row: " << weights2.rows() << "\tCol: " << weights2.cols() << endl;
	cout << weights2 << endl;

	cout << "B2\t" << "Row: " << weights_b2.rows() << "\tCol: " << weights_b2.cols() << endl;
	cout << weights_b2 << endl;
}

//保存参数到文件
void RAE::logWeights(Parameter* para)
{
	string filename;

	if(RAEType == SL)
	{
		filename = para->getPara("srcRAEWeightsLogFile");
	}
	else
	{
		filename = para->getPara("tgtRAEWeightsLogFile");
	}

	ofstream out(filename.c_str(), ios::out);

	out << "W1" << endl;
	out << weights1 << endl;

	out << "B1" << endl;
	out << weights_b1 << endl;

	out << "W2" << endl;
	out << weights2 << endl;

	out << "B2" << endl;
	out << weights_b2 << endl;

	out.close();
}

void RAE::loadWeights(Parameter* para)
{
	string filename;

	if(RAEType == SL)
	{
		filename = para->getPara("srcRAEWeightsLogFile");
	}
	else
	{
		filename = para->getPara("tgtRAEWeightsLogFile");
	}

	ifstream in(filename.c_str(), ios::in);

	bool rae_w1 = false;
	bool rae_b1 = false;
	bool rae_w2 = false;
	bool rae_b2 = false;
	int row = 0;

	string line;
	while(getline(in, line))
	{
		if(line.find("W1") == 0)
		{
			row = 0;
			rae_w1 = true;
			continue;
		}
		if(line.find("W2") == 0)
		{
			row = 0;
			rae_b1 = false;
			rae_w2 = true;
			continue;
		}
		if(line.find("B1") == 0)
		{
			row = 0;
			rae_w1 = false;
			rae_b1 = true;
			continue;
		}
		if(line.find("B2") == 0)
		{
			row = 0;
			rae_w2 = false;
			rae_b2 = true;
			continue;
		}

		if(rae_w1)
		{
			stringstream ss(line);

			for(int col = 0; col < weights1.cols(); col++)
			{
				ss >> weights1(row, col);
			}
		}
		if(rae_w2)
		{
			stringstream ss(line);

			for(int col = 0; col < weights2.cols(); col++)
			{
				ss >> weights2(row, col);
			}
		}
		if(rae_b2)
		{
			stringstream ss(line);

			for(int col = 0; col < weights_b2.cols(); col++)
			{
				ss >> weights_b2(row, col);
			}
		}
		if(rae_b1)
		{
			stringstream ss(line);

			for(int col = 0; col < weights_b1.cols(); col++)
			{
				ss >> weights_b1(row, col);
			}
		}

		row++;
	}
}

//构建RAE树
void RAE::buildTree(string bp)
{
	vector<Node*> treeNodes;
	int count = 0;
	stringstream ss(bp);
	string tmp;

	if(RAETree)
	{
		delete RAETree;
		RAETree = NULL;
	}

	while(ss >> tmp)
	{
		if(!words->isInDict(tmp))
		{
			MatrixLBFGS tmpVec = MatrixLBFGS(1, vecSize);
			tmpVec.setZero();
			words->m_words[tmp] = tmpVec;
		}

		Node* newNode = new Node(BASED_NODE, count, count, tmp, words->m_words[tmp], NULL, NULL, NULL);
		treeNodes.push_back(newNode);
	
		count++;
	}
	count--;

	if(treeNodes.size() == 1)
	{
		RAETree = new Tree(treeNodes[0]);
		return;
	}
	
	//选取Erec最小的两个based节点
	vector<lbfgsfloatval_t> v_recError;
	for(int i = 0; i < treeNodes.size()-1; i++)
	{
		RAETree = new Tree(treeNodes[i]);
		RAETree->merge(treeNodes[i+1], weights1, weights_b1, weights2, weights_b2);
		v_recError.push_back(RAETree->getRoot()->getRecError());
		delete RAETree->getRoot();
		RAETree->root = NULL;
	}

	int minNode = 0;
	lbfgsfloatval_t minRecError = v_recError[0];
	for(int i = 1; i< v_recError.size(); i++)
	{
		if(v_recError[i] < minRecError)
		{
			minNode = i;
			minRecError = v_recError[i];
		}
	}

	//建立RAE树
	RAETree = new Tree(treeNodes[minNode]);
	RAETree->merge(treeNodes[minNode+1], weights1, weights_b1, weights2, weights_b2);

	treeNodes.erase(treeNodes.begin()+minNode, treeNodes.begin()+minNode+2);

	if(treeNodes.size() == 0)
	{
		return;
	}

	//添加新节点直到覆盖整个短语
	while(treeNodes.size() != 1)
	{
		int nodePos;
		if(RAETree->getRoot()->getSpan().first == 0)
		{
			for(int i = 0; i < treeNodes.size(); i++)
			{
				if(treeNodes[i]->getSpan().first == RAETree->getRoot()->getSpan().second+1)
				{
					nodePos = i;
				}
			}
		}
		else if(RAETree->getRoot()->getSpan().second == count)
		{
			for(int i = 0; i < treeNodes.size(); i++)
			{
				if(treeNodes[i]->getSpan().second == RAETree->getRoot()->getSpan().first-1)
				{
					nodePos = i;
				}
			}
		}
		else
		{
			int pos1, pos2;

			for(int i = 0; i < treeNodes.size(); i++)
			{
				if(treeNodes[i]->getSpan().first == RAETree->getRoot()->getSpan().second+1)
				{
					pos2 = i;
				}

				if(treeNodes[i]->getSpan().second == RAETree->getRoot()->getSpan().first-1)
				{
					pos1 = i;
				}
			}

			lbfgsfloatval_t recError;

			Tree* tmpTree = new Tree(RAETree->getRoot());
			tmpTree->merge(treeNodes[pos1], weights1, weights_b1, weights2, weights_b2);
			recError = tmpTree->getRoot()->getRecError();
			delete tmpTree->getRoot();
			tmpTree->root = NULL;

			tmpTree = new Tree(RAETree->getRoot());
			tmpTree->merge(treeNodes[pos2], weights1, weights_b1, weights2, weights_b2);

			if(tmpTree->getRoot()->getRecError() < recError)
			{
				nodePos = pos2;
			}
			else
			{
				nodePos = pos1;
			}

			delete tmpTree->getRoot();
			tmpTree->root = NULL;
		}

		RAETree->merge(treeNodes[nodePos], weights1, weights_b1, weights2, weights_b2);
		treeNodes.erase(treeNodes.begin()+nodePos);
	}
	RAETree->merge(treeNodes[0], weights1, weights_b1, weights2, weights_b2);
}

int RAE::getVecSize()
{
	return vecSize;
}

RAE* RAE::copy()
{	
	RAE* newRAE = new RAE(this->vecSize);
	newRAE->vecSize = this->vecSize;
	newRAE->words = this->words;
	
	newRAE->RAETree = NULL;
	newRAE->weights1 = this->weights1;
	newRAE->weights_b1 = this->weights_b1;
	newRAE->weights2 = this->weights2;
	newRAE->weights_b2 = this->weights_b2;

	return newRAE;
}

RAE::~RAE()
{
	delete RAETree;
	RAETree = NULL;
	trainingData.clear();
}

lbfgsfloatval_t RAE::decay()
{
	lbfgsfloatval_t val = 0;

	for(int row = 0; row < weights1.rows(); row++)
	{
		for(int col = 0; col < weights1.cols(); col++)
		{
			val += pow(weights1(row, col), 2)/2;
			val += pow(weights2(col, row), 2)/2;
		}
	}

	for(int col = 0; col < weights_b1.cols(); col++)
	{
		val += pow(weights_b1(0, col), 2)/2;
	}

	for(int col = 0; col < weights_b2.cols(); col++)
	{
		val += pow(weights_b2(0, col), 2)/2;
	}

	return val;
}

void RAE::trainRecError()
{
	Node* tmpNode = this->RAETree->getRoot();

	//更新w2, b2
	while(tmpNode->getNodeType() != BASED_NODE)
	{
		MatrixLBFGS c = concatMatrix(tmpNode->getLeftChildNode()->getVector(),tmpNode->getRightChildNode()->getVector());
		MatrixLBFGS cRec = concatMatrix(tmpNode->leftReconst,tmpNode->rightReconst);

		for(int row = 0; row < weights2.rows(); row++)
		{
			lbfgsfloatval_t result = (cRec(0, row)-c(0, row)) * (1-pow(cRec(0, row), 2));
			for(int col = 0; col < weights2.cols(); col++)
			{
				delWeight2(row, col) = delWeight2(row,col) + ALPHA * result * tmpNode->getVector()(0, col);
			}
			delWeight2_b(0, row) = ALPHA * result + delWeight2_b(0, row);
		}

		tmpNode = tmpNode->getLeftChildNode();
	}

	//仅对每一对重构中的权重求导
	tmpNode = this->RAETree->getRoot();

	while(tmpNode->getNodeType() != BASED_NODE)
	{
		MatrixLBFGS c = concatMatrix(tmpNode->getLeftChildNode()->getVector(),tmpNode->getRightChildNode()->getVector());
		MatrixLBFGS cRec = concatMatrix(tmpNode->leftReconst,tmpNode->rightReconst);
		MatrixLBFGS tmpDelWb = MatrixLBFGS(1, 2*vecSize);

		for(int row = 0; row < weights2.rows(); row++)
		{
			lbfgsfloatval_t result = (cRec(0, row)-c(0, row)) * (1-pow(cRec(0, row), 2));
			tmpDelWb(0, row) = result;
		}

		tmpDelWb = tmpDelWb*weights2;

		for(int col = 0; col < tmpDelWb.cols(); col++)
		{
			tmpDelWb(0, col) *= (1-pow(tmpNode->getVector()(0, col), 2));
		}

		for(int row = 0; row < delWeight1.rows(); row++)
		{
			delWeight1_b(0, row) = delWeight1_b(0, row)+ ALPHA * tmpDelWb(0, row);
			for(int col = 0; col < delWeight1.cols(); col++)
			{
				delWeight1(row, col) = delWeight1(row, col) + ALPHA * tmpDelWb(0, row) * c(0, col);
			}
		}

		tmpNode = tmpNode->getLeftChildNode();
	}

	delWeight1 += ZETA*weights1;
	delWeight2 += ZETA*weights2;
}

//读取训练数据
void RAE::loadTrainingData()
{
	trainingData.clear();
	string dataFile;
	string domainName;
	string domainLine = para->getPara("DomainList");
	bool isDev = atoi(para->getPara("IsDev").c_str());
	bool isTrain = atoi(para->getPara("IsTrain").c_str());
	bool isTest = atoi(para->getPara("IsTest").c_str());

	stringstream ss(domainLine);
	int count = 0;
	while(ss >> domainName)
	{
		if(count == atoi( para->getPara("THREAD_NUM").c_str() ))
		{
			break;
		}

		if(isDev)
		{
			dataFile = para->getPara(domainName + "DevDataFile");
		}
		else if(isTrain)
		{
			dataFile = para->getPara(domainName + "DataFile");
		}

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

			if(RAEType == SL)
			{
				trainingData.push_back(m_tmp["ct1"]);
				trainingData.push_back(m_tmp["ct2"]);
			}
			else
			{
				trainingData.push_back(m_tmp["et1"]);
				trainingData.push_back(m_tmp["et2"]);
			}
		}

		in.close();

		count++;
	}
}

void RAE::training()
{
	x = lbfgs_malloc(getRAEWeightSize());
	Map<MatrixLBFGS>(x, getRAEWeightSize(), 1).setRandom();
	lbfgs_parameter_t param;
	iterTimes = atoi(para->getPara("IterationTime").c_str());

	loadTrainingData();
	lbfgs_parameter_init(&param);
	param.max_iterations = iterTimes;

	lbfgsfloatval_t fx = 0;
	int ret;

	ret = lbfgs(getRAEWeightSize(), x, &fx, RAELBFGS::evaluate, RAELBFGS::progress, this, &param);

	logWeights(para);
	trainingData.clear();
	lbfgs_free(x);
}

void RAE::update(lbfgsfloatval_t* g)
{
	Map<MatrixLBFGS> g_Weights1(g, weights1.rows(), weights1.cols());
	Map<MatrixLBFGS> g_Weights_b1(g+weights1.rows()*weights1.cols(), weights_b1.rows(), weights_b1.cols());
	Map<MatrixLBFGS> g_Weights2(g+weights1.rows()*weights1.cols()+
		weights_b1.rows()*weights_b1.cols(),
		weights2.rows(), weights2.cols());
	Map<MatrixLBFGS> g_Weights_b2(g+weights1.rows()*weights1.cols()+
		weights_b1.rows()*weights_b1.cols()+
		weights2.rows()*weights2.cols(),
		weights_b2.rows(), weights_b2.cols());

	g_Weights1 += delWeight1;
	g_Weights_b1 += delWeight1_b;
	g_Weights2 += delWeight2;
	g_Weights_b2 += delWeight2_b;
}

lbfgsfloatval_t RAE::_training(lbfgsfloatval_t* g)
{
	lbfgsfloatval_t error = 0;

	for(int i = 0; i < trainingData.size(); i++)
	{
		//获取实例
		buildTree(trainingData[i]);	

		error += loss();
		
		//对rae求导
		trainRecError();

		update(g);

		delete RAETree;
		RAETree = NULL;
	}

	return error;
}

void RAE::updateWeights(const lbfgsfloatval_t* x)
{
	lbfgsfloatval_t* cX = const_cast<lbfgsfloatval_t*>(x);

	weights1 = Map<MatrixLBFGS>(cX, vecSize, 2*vecSize);
	weights_b1 = Map<MatrixLBFGS>(cX + 2*vecSize*vecSize, 1, vecSize);
	weights2 = Map<MatrixLBFGS>(cX + 2*vecSize*vecSize+vecSize, 2*vecSize, vecSize);
	weights_b2 = Map<MatrixLBFGS>(cX + 2*vecSize*vecSize+vecSize+2*vecSize*vecSize, 1, 2*vecSize);
}

lbfgsfloatval_t RAE::_evaluate(const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step)
{
	lbfgsfloatval_t fx = 0;

	int RAEThreadNum = atoi(para->getPara("RAEThreadNum").c_str());
	RAEThreadPara* threadpara = new RAEThreadPara[RAEThreadNum];
	int batchsize = trainingData.size() / RAEThreadNum;
	updateWeights(x);

	for(int i = 0; i < RAEThreadNum; i++)
	{
		threadpara[i].cRAE = this->copy();
		threadpara[i].g = lbfgs_malloc(getRAEWeightSize());
		if(i == RAEThreadNum-1)
		{
			threadpara[i].cRAE->trainingData.assign(trainingData.begin()+i*batchsize, trainingData.end());
			threadpara[i].instance_num = trainingData.size()%batchsize;
		}
		else
		{
			threadpara[i].cRAE->trainingData.assign(trainingData.begin()+i*batchsize, trainingData.begin()+(i+1)*batchsize);
			threadpara[i].instance_num = batchsize;
		}
	}
	pthread_t* pt = new pthread_t[RAEThreadNum];
	for (int a = 0; a < RAEThreadNum; a++) pthread_create(&pt[a], NULL, RAELBFGS::deepThread, (void *)(threadpara + a));
	for (int a = 0; a < RAEThreadNum; a++) pthread_join(pt[a], NULL);

	for(int i = 0; i < RAEThreadNum; i++)
	{
		fx += threadpara[i].lossVal;
		for(int elem = 0; elem < getRAEWeightSize(); elem++)
		{
			g[elem] += threadpara[i].g[elem];
		}
	}

	fx /= trainingData.size();
	for(int elem = 0; elem < getRAEWeightSize(); elem++)
	{
		g[elem] /= trainingData.size();
	}

	delete pt;
	pt = NULL;
	
	for(int i  = 0; i < RAEThreadNum; i++)
	{
		lbfgs_free(threadpara[i].g);
		delete threadpara[i].cRAE;
	}
	delete threadpara;
	threadpara = NULL;

	return fx;
}

int RAE::_progress(const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls)
{
	ofstream out("./log/RAE/RAE.log", ios::app);

	out << "Iteration of RAE: " << k << endl;
	out << "Loss Value: " << fx << endl;

	out.close();

	return 0;
}
