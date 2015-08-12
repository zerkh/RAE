#include "RAETree.h"

Node* Tree::getRoot()
{
	return root;
}

Tree::Tree(Node* root)
{
	this->root = root;
}

void Tree::showTree()
{
	Node* leftNode;
	Node* rightNode;

	if(root == NULL)
	{
		cout << "NULL" << endl;
		return;
	}

	leftNode = root->getLeftChildNode();
	rightNode = root->getRightChildNode();

	int count = 0;

	cout << root->getSpan().first << "," << root->getSpan().second << "\t" << root->getWord() << " " << root->getNodeType() << endl;

	while(leftNode != NULL)
	{
		count++;
		cout << count << endl;
		cout << leftNode->getSpan().first << "," << leftNode->getSpan().second << "\t" << leftNode->getWord() << " " << leftNode->getNodeType() << endl;
		cout << rightNode->getSpan().first << "," << rightNode->getSpan().second << "\t" << rightNode->getWord() << " " << rightNode->getNodeType() << endl;

		leftNode = leftNode->getLeftChildNode();
		rightNode = leftNode->getRightChildNode();
	}
}

void Tree::merge(Node* newNode, Vector* w1, Vector* b1, Vector* w2, Vector* b2)
{
	Vector* tmpCon = root->getVector()->concat(newNode->getVector());
	Vector* tmpMul = tmpCon->multiply(w1, true);
	Vector* parent = tmpMul->add(b1);
	delete tmpCon;
	delete tmpMul;

	Node* pNode = NULL;
	if(root->getSpan().second < newNode->getSpan().first)
	{
		pNode = new Node(COMBINED_NODE, root->getSpan().first, newNode->getSpan().second, root->getWord() + " " +newNode->getWord(), parent, NULL, root, newNode);
		root->setParentNode(pNode);
		newNode->setParentNode(pNode);
	}
	else
	{
		pNode = new Node(COMBINED_NODE, newNode->getSpan().first, root->getSpan().second, newNode->getWord() + " " +root->getWord(), parent, NULL, root, newNode);
		root->setParentNode(pNode);
		newNode->setParentNode(pNode);
	}

	Vector* rec = parent->multiply(w2, true)->add(b2);

	pNode->leftReconst = new Vector(rec->getRow(), rec->getCol()/2);

	pNode->rightReconst = new Vector(rec->getRow(), rec->getCol()/2);

	for(int i = 0; i < pNode->leftReconst->getCol(); i++)
	{
		pNode->leftReconst->setValue(0, i, rec->getValue(0, i));
		pNode->rightReconst->setValue(0, i, rec->getValue(0, i+pNode->leftReconst->getCol()));
	}

	root = pNode;
}
