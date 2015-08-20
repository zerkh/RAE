#include "RAE.h"
#include "Util.h"

RAE::RAE(Parameter* para, WordVec* words)
{
	vecSize = atoi(para->getPara("WordVecSize").c_str());
	this->words = words;

	RAETree = NULL;
	weights1 = MatrixXd(vecSize, vecSize*2);
	weights2 = MatrixXd(vecSize*2, vecSize);
	weights_b1 = MatrixXd(1, vecSize);
	weights_b2 = MatrixXd(1, vecSize*2);

	delWeight1 = MatrixXd(weights1.rows(), weights1.cols());
	delWeight1_b = MatrixXd(weights_b1.rows(), weights_b1.cols());
	delWeight2 = MatrixXd(weights2.rows(), weights2.cols());
	delWeight2_b = MatrixXd(weights_b2.rows(), weights_b2.cols());

	delWeight1.setZero();
	delWeight1_b.setZero();
	delWeight2.setZero();
	delWeight2_b.setZero();
	
	weights1.setRandom();
	weights2.setRandom();
	weights_b1.setRandom();
	weights_b2.setRandom();	
}

RAE::RAE(int size)
{
	RAETree = NULL;
	this->vecSize = size;
	delWeight1 = MatrixXd(vecSize, vecSize*2);
	delWeight1_b = MatrixXd(1, vecSize);
	delWeight2 = MatrixXd(vecSize*2, vecSize);
	delWeight2_b = MatrixXd(1, vecSize*2);

	delWeight1.setZero();
	delWeight1_b.setZero();
	delWeight2.setZero();
	delWeight2_b.setZero();
}

double RAE::loss()
{
	Node* tmpNode = this->RAETree->getRoot();
	double val = 0;

	while(tmpNode->getNodeType() != BASED_NODE)
	{
		val += tmpNode->getRecError();

		tmpNode = tmpNode->getLeftChildNode();
	}

	return val;
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
	string filename = para->getPara("RAEWeightsLogFile");

	ofstream out(filename.c_str(), ios::out);

	out << "W1" << endl;
	out << weights1 << endl;

	out << "B1" << endl;
	out << weights_b1 << endl;

	out << "W2" << endl;
	out << weights2 << endl;

	out << "B2" << endl;
	out << weights_b2 << endl;
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
			MatrixXd tmpVec = MatrixXd(1, vecSize);
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
	vector<double> v_recError;
	for(int i = 0; i < treeNodes.size()-1; i++)
	{
		RAETree = new Tree(treeNodes[i]);
		RAETree->merge(treeNodes[i+1], weights1, weights_b1, weights2, weights_b2);
		v_recError.push_back(RAETree->getRoot()->getRecError());
		delete RAETree->getRoot();
	}

	int minNode = 0;
	double minRecError = v_recError[0];
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

			double recError;

			Tree* tmpTree = new Tree(RAETree->getRoot());
			tmpTree->merge(treeNodes[pos1], weights1, weights_b1, weights2, weights_b2);
			recError = tmpTree->getRoot()->getRecError();
			delete tmpTree->getRoot();

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
	Node* n1 = RAETree->getRoot();
	Node* n2 = RAETree->getRoot();

	while(n2 != NULL)
	{
		n2 = n1->getRightChildNode();
		if(n2 != NULL)
		{	
			delete n2;
		}
		
		n2 = n1->getLeftChildNode();
		delete n1;
		
		n1 = n2;
	}

}

double RAE::decay()
{
	double val = 0;

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
		MatrixXd c = concatMatrix(tmpNode->getLeftChildNode()->getVector(),tmpNode->getRightChildNode()->getVector());
		MatrixXd cRec = concatMatrix(tmpNode->leftReconst,tmpNode->rightReconst);
		for(int row = 0; row < weights2.rows(); row++)
		{
			double result = (cRec(0, row)-c(0, row)) * (1-pow(cRec(0, row), 2));
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
		MatrixXd c = concatMatrix(tmpNode->getLeftChildNode()->getVector(),tmpNode->getRightChildNode()->getVector());
		MatrixXd cRec = concatMatrix(tmpNode->leftReconst,tmpNode->rightReconst);
		MatrixXd tmpDelWb = MatrixXd(1, 2*vecSize);

		for(int row = 0; row < weights2.rows(); row++)
		{
			double result = (cRec(0, row)-c(0, row)) * (1-pow(cRec(0, row), 2));
			tmpDelWb(0, row) = result;
		}

		tmpDelWb = tmpDelWb*weights2;
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
}
