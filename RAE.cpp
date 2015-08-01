#include "RAE.h"

RAE::RAE(Parameter* para, WordVec* words)
{
	vecSize = atoi(para->getPara("WordVecSize").c_str());
	this->words = words;

	weights1 = new Vector(vecSize, vecSize*2);
	weights2 = new Vector(vecSize*2, vecSize);
	weights_b1 = new Vector(1, vecSize);
	weights_b2 = new Vector(1, vecSize*2);

	delWeight1 = new Vector(weights1->getRow(), weights1->getCol());
	delWeight1_b = new Vector(weights_b1->getRow(), weights_b1->getCol());
	delWeight2 = new Vector(weights2->getRow(), weights2->getCol());
	delWeight2_b = new Vector(weights_b2->getRow(), weights_b2->getCol());

	delWeight1->setToZeros();
	delWeight1_b->setToZeros();
	delWeight2->setToZeros();
	delWeight2_b->setToZeros();
	
	weights1->randInitVector();
	weights2->randInitVector();
	weights_b1->randInitVector();
	weights_b2->randInitVector();	
}

RAE::RAE(int size)
{
	this->vecSize = size;
	delWeight1 = new Vector(vecSize, vecSize*2);
	delWeight1_b = new Vector(1, vecSize);
	delWeight2 = new Vector(vecSize*2, vecSize);
	delWeight2_b = new Vector(1, vecSize*2);
	
	weights1 = NULL;
	weights2 = NULL;
	weights_b1 = NULL;
	weights_b2 = NULL;	

	delWeight1->setToZeros();
	delWeight1_b->setToZeros();
	delWeight2->setToZeros();
	delWeight2_b->setToZeros();
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
	cout << "W1\t" << "Row: " << weights1->getRow() << "\tCol: " << weights1->getCol() << endl;
	for(int row = 0; row < weights1->getRow(); row++)
	{
		for(int col = 0; col < weights1->getCol(); col++)
		{
			cout << weights1->getValue(row, col) << " ";
		}

		cout << endl;
	}

	cout << "B1\t" << "Row: " << weights_b1->getRow() << "\tCol: " << weights_b1->getCol() << endl;
	for(int row = 0; row < weights_b1->getRow(); row++)
	{
		for(int col = 0; col < weights_b1->getCol(); col++)
		{
			cout << weights_b1->getValue(row, col) << " ";
		}

		cout << endl;
	}

	cout << "W2\t" << "Row: " << weights2->getRow() << "\tCol: " << weights2->getCol() << endl;
	for(int row = 0; row < weights2->getRow(); row++)
	{
		for(int col = 0; col < weights2->getCol(); col++)
		{
			cout << weights2->getValue(row, col) << " ";
		}

		cout << endl;
	}

	cout << "B2\t" << "Row: " << weights_b2->getRow() << "\tCol: " << weights_b2->getCol() << endl;
	for(int row = 0; row < weights_b2->getRow(); row++)
	{
		for(int col = 0; col < weights_b2->getCol(); col++)
		{
			cout << weights_b2->getValue(row, col) << " ";
		}

		cout << endl;
	}
}

//保存参数到文件
void RAE::logWeights(Parameter* para)
{
	string filename = para->getPara("RAEWeightsLogFile");

	ofstream out(filename.c_str(), ios::out);

	out << "W1" << endl;
	for(int row = 0; row < weights1->getRow(); row++)
	{
		for(int col = 0; col < weights1->getCol(); col++)
		{
			out << weights1->getValue(row, col) << " ";
		}

		out << endl;
	}

	out << "B1" << endl;
	for(int row = 0; row < weights_b1->getRow(); row++)
	{
		for(int col = 0; col < weights_b1->getCol(); col++)
		{
			out << weights_b1->getValue(row, col) << " ";
		}

		out << endl;
	}

	out << "W2" << endl;
	for(int row = 0; row < weights2->getRow(); row++)
	{
		for(int col = 0; col < weights2->getCol(); col++)
		{
			out << weights2->getValue(row, col) << " ";
		}

		out << endl;
	}

	out << "B2" << endl;
	for(int row = 0; row < weights_b2->getRow(); row++)
	{
		for(int col = 0; col < weights_b2->getCol(); col++)
		{
			out << weights_b2->getValue(row, col) << " ";
		}

		out << endl;
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
			Vector* tmpVec = new Vector(1, vecSize);
			tmpVec->setToZeros();
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
		delete RAETree;
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

			delete tmpTree;
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
	
	newRAE->weights1 = this->weights1->copy();
	newRAE->weights_b1 = this->weights_b1->copy();
	newRAE->weights2 = this->weights2->copy();
	newRAE->weights_b2 = this->weights_b2->copy();

	return newRAE;
}

RAE::~RAE()
{
	delete weights1;
	delete weights_b1;
	delete weights2;
	delete weights_b2;

	delete delWeight1;
	delete delWeight1_b;
	delete delWeight2;
	delete delWeight2_b;

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

	for(int row = 0; row < weights1->getRow(); row++)
	{
		for(int col = 0; col < weights1->getCol(); col++)
		{
			val += pow(weights1->getValue(row, col), 2)/2;
			val += pow(weights2->getValue(col, row), 2)/2;
		}
	}

	for(int col = 0; col < weights_b1->getCol(); col++)
	{
		val += pow(weights_b1->getValue(0, col), 2)/2;
	}

	for(int col = 0; col < weights_b2->getCol(); col++)
	{
		val += pow(weights_b2->getValue(0, col), 2)/2;
	}

	return val;
}

void RAE::trainRecError()
{
	Node* tmpNode = this->RAETree->getRoot();

	//更新w2, b2
	while(tmpNode->getNodeType() != BASED_NODE)
	{
		Vector* c = tmpNode->getLeftChildNode()->getVector()->concat(tmpNode->getRightChildNode()->getVector());
		Vector* cRec = tmpNode->leftReconst->concat(tmpNode->rightReconst);
		for(int row = 0; row < weights2->getRow(); row++)
		{
			double result = (cRec->getValue(0, row)-c->getValue(0, row)) * (1-pow(cRec->getValue(0, row), 2));
			for(int col = 0; col < weights2->getCol(); col++)
			{
				delWeight2->setValue(row, col, delWeight2->getValue(row,col) + ALPHA * result*tmpNode->getVector()->getValue(0, col));
			}
			delWeight2_b->setValue(0, row, ALPHA * result + delWeight2_b->getValue(0, row));
		}

		tmpNode = tmpNode->getLeftChildNode();

		delete c;
		delete cRec;
	}

	//仅对每一对重构中的权重求导
	tmpNode = this->RAETree->getRoot();
	Vector* tmpDelWb = new Vector(1, 2*vecSize);

	while(tmpNode->getNodeType() != BASED_NODE)
	{
		Vector* c = tmpNode->getLeftChildNode()->getVector()->concat(tmpNode->getRightChildNode()->getVector());
		Vector* cRec = tmpNode->leftReconst->concat(tmpNode->rightReconst);
		for(int row = 0; row < weights2->getRow(); row++)
		{
			double result = (cRec->getValue(0, row)-c->getValue(0, row)) * (1-pow(cRec->getValue(0, row), 2));
			tmpDelWb->setValue(0, row, result);
		}

		tmpDelWb = tmpDelWb->multiply(weights2, false);
		for(int row = 0; row < delWeight1->getRow(); row++)
		{
			delWeight1_b->setValue(0, row, delWeight1_b->getValue(0, row)+ ALPHA * tmpDelWb->getValue(0, row));
			for(int col = 0; col < delWeight1->getCol(); col++)
			{
				delWeight1->setValue(row, col, delWeight1->getValue(row, col) + ALPHA * tmpDelWb->getValue(0, row) * c->getValue(0, col));
			}
		}

		tmpNode = tmpNode->getLeftChildNode();
		delete c;
		delete cRec;
	}

	delete tmpDelWb;
}
