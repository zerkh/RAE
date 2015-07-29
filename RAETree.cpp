#include "RAETree.h"

Node* Tree::getRoot()
{
	return root;
}

Tree::Tree(Node* root)
{
	this->root = root;
}

void Tree::merge(Node* newNode, Vector* w1, Vector* b1, Vector* w2, Vector* b2)
{
	Vector* parent = root->getVector()->concat(newNode->getVector())->multiply(w1, true)->add(b1);

	Node* pNode;
	if(root->getSpan().second < newNode->getSpan().first)
	{
		pNode = new Node(COMBINED_NODE, root->getSpan().first, newNode->getSpan().second, root->getWord() + " " +newNode->getWord(), parent, NULL, root, newNode);
	}
	else
	{
		pNode = new Node(COMBINED_NODE, newNode->getSpan().first, root->getSpan().second, newNode->getWord() + " " +root->getWord(), parent, NULL, root, newNode);
	}

	Vector* rec = parent->multiply(w2, true)->add(b2);

	pNode->leftReconst = new Vector(rec->getRow(), rec->getCol()/2);

	pNode->rightReconst = new Vector(rec->getRow(), rec->getCol()/2);

	for(int i = 0; i < pNode->leftReconst->getCol(); i++)
	{
		pNode->leftReconst->setValue(0, i, rec->getValue(0, i));
		pNode->rightReconst->setValue(0, i, rec->getValue(0, i+pNode->rightReconst->getCol()));
	}

	root = pNode;
}