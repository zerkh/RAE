#include "RAETree.h"
#include "Util.h"

Node* Tree::getRoot()
{
	return root;
}

Tree::Tree(Node* root)
{
	this->root = root;
}

void Tree::merge(Node* newNode, MatrixXd w1, MatrixXd b1, MatrixXd w2, MatrixXd b2)
{
	MatrixXd tmpCon = concatMatrix(root->getVector(),newNode->getVector());
	MatrixXd tmpMul = tmpCon * w1.transpose();
	MatrixXd parent = tmpMul + b1;

	Node* pNode;
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

	MatrixXd rec = parent * w2.transpose() + b2;

	pNode->leftReconst = MatrixXd(rec.rows(), rec.cols()/2);

	pNode->rightReconst = MatrixXd(rec.rows(), rec.cols()/2);

	for(int i = 0; i < pNode->leftReconst.cols(); i++)
	{
		pNode->leftReconst(0, i) = rec(0, i);
		pNode->rightReconst(0, i) = rec(0, i+pNode->leftReconst.cols());
	}

	root = pNode;
}
