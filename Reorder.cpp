#include "Reorder.h"

ReorderModel::ReorderModel(Parameter* para, WordVec* words)
{
	vecSize = atoi(para->getPara("WordVecSize").c_str());

	rae1 = new RAE(para, words);
	rae2 = new RAE(para, words);

	weights = new Vector(2, vecSize);
	weights_b = new Vector(1, 2);

	delWeight = new Vector(weights->getRow(), weights->getCol());
	delWeight_b = new Vector(weights_b->getRow(), weights_b->getCol());

	weights->randInitVector();
	weights_b->randInitVector();
}

double ReorderModel::decay()
{
	double val = 0;

	for(int row = 0; row < weights->getRow(); row++)
	{
		for (int col = 0; col < weights->getCol(); col++)
		{
			val += pow(weights->getValue(row, col), 2) / 2;
		}

		val += pow(weights_b->getValue(0, row), 2) / 2;
	}

	return val;
}

void ReorderModel::softmax()
{
	outputLayer = rae1->RAETree->getRoot()->getVector()->concat(rae2->RAETree->getRoot()->getVector())->multiply(weights, true)->add(weights_b);

	double result;

	result = exp(outputLayer->getValue(0, 0))/(exp(outputLayer->getValue(0, 0)) + exp(outputLayer->getValue(0, 1)));

	Vector* soft = new Vector(1, 2);

	soft->setValue(0, 0, result);
	soft->setValue(0, 1, 1-result);

	softmaxLayer = soft;
}

void ReorderModel::getData(string bp1, string bp2)
{
	rae1->buildTree(bp1);

	rae2->buildTree(bp2);

	softmax();
}

void ReorderModel::trainRM(Vector* y, bool isSoftmax)
{
	double p;
	if(isSoftmax)
	{
		p = GAMMA;
	}
	else
	{
		p = BETA;
	}

	Vector* theta = new Vector(delWeight_b->getRow(), delWeight_b->getCol());

	//对W和Wb求导
	for(int row = 0; row < weights->getRow(); row)
	{
		double result;
		result = (softmaxLayer->getValue(0, row) - y->getValue(0, row)) * (exp(outputLayer->getValue(0, 0))*exp(outputLayer->getValue(0, 1))/(exp(outputLayer->getValue(0, 0))+exp(outputLayer->getValue(0, 1))));
		
		for(int col = 0; col < weights->getCol(); col++)
		{
			Vector* tmpX = rae1->RAETree->getRoot()->getVector()->concat(rae2->RAETree->getRoot()->getVector());
			delWeight->setValue(row, col, delWeight->getValue(row, col) + p * result * tmpX->getValue(0, col));
		}

		delWeight_b->setValue(0, row, delWeight_b->getValue(0, row) + p * result);
		theta->setValue(0, row, result);
	}

	//对W1和Wb1求导
	Node* preNode1 = rae1->RAETree->getRoot();
	Node* preNode2 = rae2->RAETree->getRoot();

	theta = theta->multiply(weights, false);
	Vector* theta1 = new Vector(theta->getRow(), theta->getCol()/2);
	Vector* theta2 = new Vector(theta->getRow(), theta->getCol()/2);

	for(int i = 0; i < theta->getCol(); i++)
	{
		theta1->setValue(0, i, theta->getValue(0, i));
		theta2->setValue(0, i, theta->getValue(0, i+theta1->getCol()));
	}

	while(preNode1->getNodeType() != BASED_NODE)
	{
		for(int row = 0; row < vecSize; row++)
		{
			for(int col = 0; col < vecSize*2; col++)
			{
				if(col < vecSize)
				{
					delWeight->setValue(row, col, delWeight->getValue(row, col) + p * theta1->getValue(0, row)*preNode1->getLeftChildNode()->getVector()->getValue(0, col));
				}
				else
				{
					delWeight->setValue(row, col, delWeight->getValue(row, col) + p * theta1->getValue(0, row)*preNode1->getRightChildNode()->getVector()->getValue(0, col-vecSize));
				}
			}

			delWeight_b->setValue(0, row, delWeight_b->getValue(0, row) + p * theta1->getValue(0, row));
		}

		theta1 = theta1->multiply(rae1->weights1, false);
		Vector* tmpTheta = new Vector(theta->getRow(), theta->getCol()/2);

		for(int i = 0; i < theta1->getCol(); i++)
		{
			tmpTheta->setValue(0, i, theta1->getValue(0, i));
		}

		theta1 = tmpTheta;
		preNode1 = preNode1->getLeftChildNode();
	}

	while(preNode2->getNodeType() != BASED_NODE)
	{
		for(int row = 0; row < vecSize; row++)
		{
			for(int col = 0; col < vecSize*2; col++)
			{
				if(col < vecSize)
				{
					delWeight->setValue(row, col, delWeight->getValue(row, col) + p * theta2->getValue(0, row)*preNode2->getLeftChildNode()->getVector()->getValue(0, col));
				}
				else
				{
					delWeight->setValue(row, col, delWeight->getValue(row, col) + p * theta2->getValue(0, row)*preNode2->getRightChildNode()->getVector()->getValue(0, col-vecSize));
				}
			}

			delWeight_b->setValue(0, row, delWeight_b->getValue(0, row) + p * theta2->getValue(0, row));
		}

		theta2 = theta2->multiply(rae1->weights1, false);
		Vector* tmpTheta = new Vector(theta->getRow(), theta->getCol()/2);

		for(int i = 0; i < theta2->getCol(); i++)
		{
			tmpTheta->setValue(0, i, theta2->getValue(0, i));
		}

		theta2 = tmpTheta;
		preNode2 = preNode2->getLeftChildNode();
	}

	delete theta;
	delete theta1;
	delete theta2;
}