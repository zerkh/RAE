#ifndef RAETREE
#define RAETREE
#include "Constdef.h"
#include "Vec.h"

class Node
{
private:
	Node* parent;					//父节点
	Node* leftChild;				//左子节点(通常为combined节点)
	Node* rightChild;				//右子节点(通常为based节点)
	int nodeType;					//节点类型
	Vector* vec;					//存储向量值
	string word;
	span sp;

public:
	Vector* leftReconst;				//左重建节点
	Vector* rightReconst;				//右重建节点

	Node(int nodeType, int start, int end, string word, Vector* vec, Node* parent, Node* leftChild, Node* rightChild)
	{
		sp = make_pair(start, end);
		this->word = word;
		this->nodeType = nodeType;
		this->vec = vec;
		this->parent = parent;
		this->leftChild = leftChild;
		this->rightChild = rightChild;
		this->leftReconst = NULL;
		this->rightReconst = NULL;
	}

	~Node()
	{
		rightChild->parent = NULL;
		leftChild->parent = NULL;
		
		if(!leftReconst)
		{
			delete leftReconst;
			delete rightReconst;
		}
	}

	double getRecError()
	{
		double RecError = 0;

		if(leftChild == NULL && rightChild == NULL)
		{
			return RecError;
		}

		for(int i = 0; i < leftReconst->getCol(); i++)
		{
			RecError += pow(leftChild->getVector()->getValue(0, i) - leftReconst->getValue(0, i), 2) + pow(rightChild->getVector()->getValue(0, i) - rightReconst->getValue(0, i), 2);
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

	Vector* getVector()
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

	void setVector(Vector* vec)
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
		while(n2->getNodeType() != BASED_NODE)
		{
			n2 = n1->getLeftChildNode();
			delete n1;
			n1 = n2;
		}
	}

	Node* getRoot();

	void merge(Node* newNode, Vector* w1, Vector* b1, Vector* w2, Vector* b2);
};

#endif
