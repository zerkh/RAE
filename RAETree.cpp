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
	MatrixXd parent = concatMatrix(root->getVector(),newNode->getVector()) * w1.transpose() + b1;

	parent = tanh(parent);

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

	rec = tanh(rec);

	pNode->leftReconst = MatrixXd(rec.rows(), rec.cols()/2);

	pNode->rightReconst = MatrixXd(rec.rows(), rec.cols()/2);

	for(int i = 0; i < pNode->leftReconst.cols(); i++)
	{
		pNode->leftReconst(0, i) = rec(0, i);
		pNode->rightReconst(0, i) = rec(0, i+pNode->leftReconst.cols());
	}

	root = pNode;
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

		rightNode = leftNode->getRightChildNode();
		leftNode = leftNode->getLeftChildNode();
	}
}