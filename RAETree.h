#ifndef RAETREE
#define RAETREE

#include "Util.h"
#include "Constdef.h"
#include "Eigen/Core"

using namespace Eigen;

class Node
{
private:
	Node* parent;					//父节点
	Node* leftChild;				//左子节点(通常为combined节点)
	Node* rightChild;				//右子节点(通常为based节点)
	int nodeType;					//节点类型
	MatrixXd vec;					//存储向量值
	string word;
	span sp;

public:
	MatrixXd leftReconst;				//左重建节点
	MatrixXd rightReconst;				//右重建节点

	Node(int nodeType, int start, int end, string word, MatrixXd vec, Node* parent, Node* leftChild, Node* rightChild)
	{
		sp = make_pair(start, end);
		this->word = word;
		this->nodeType = nodeType;
		this->vec = vec;
		this->parent = parent;
		this->leftChild = leftChild;
		this->rightChild = rightChild;
	}

	~Node()
	{
	}

	double getRecError()
	{
		double RecError = 0;

		if(leftChild == NULL && rightChild == NULL)
		{
			return RecError;
		}

		for(int i = 0; i < leftReconst.cols(); i++)
		{
			RecError += pow(leftChild->getVector()(0, i) - leftReconst(0, i), 2) + pow(rightChild->getVector()(0, i) - rightReconst(0, i), 2);
		}

		return RecError / 2;
	}

	span getSpan()
	{
		return sp;
	}

	string getWord()
	{
		return word;
	}

	Node* getParentNode()
	{
		return parent;
	}

	Node* getLeftChildNode()
	{
		return leftChild;
	}

	Node* getRightChildNode()
	{
		return rightChild;
	}

	MatrixXd getVector()
	{
		return vec;
	}

	int getNodeType()
	{
		return nodeType;
	}

	void setParentNode(Node* parent)
	{
		this->parent = parent;
	}

	void setLeftChildNode(Node* leftChild)
	{
		this->leftChild = leftChild;
	}

	void setRightChildNode(Node* rightChild)
	{
		this->rightChild = rightChild;
	}

	void setNodeType(int nodeType)
	{
		this->nodeType = nodeType;
	}

	void setVector(MatrixXd vec)
	{
		this->vec = vec;
	}
};

class Tree
{
	Node* root;

public:
	Tree(Node* root);

	~Tree()
	{
		Node* n1;
		Node* n2;

		n1 = root;
		n2 = root;
		while(n2 != NULL)
		{
			n2 = n1->getRightChildNode();
			delete n2;
			n2 = NULL;
			n2 = n1->getLeftChildNode();
			delete n1;
			n1 = NULL;
			n1 = n2;
		}

		root = NULL;
	}

	Node* getRoot();

	void merge(Node* newNode, MatrixXd w1, MatrixXd b1, MatrixXd w2, MatrixXd b2);

	void showTree();
};

#endif
