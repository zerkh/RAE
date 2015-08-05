#include "Reorder.h"
#include <limits>

ReorderModel::ReorderModel(Parameter* para, WordVec* words)
{
	vecSize = atoi(para->getPara("WordVecSize").c_str());

	rae = new RAE(para, words);
	rae1 = rae->copy();
	rae2 = rae->copy();

	weights = MatrixXd(2, vecSize*2);
	weights_b = MatrixXd(1, 2);

	delWeight = MatrixXd(weights.rows(), weights.cols());
	delWeight_b = MatrixXd(weights_b.rows(), weights_b.cols());

	delWeight.setZero();
	delWeight_b.setZero();

	weights.setRandom();
	weights_b.setRandom();

	outputLayer = MatrixXd(1, 2);
	softmaxLayer = MatrixXd(1, 2);
}

double ReorderModel::decay()
{
	double val = 0;

	for(int row = 0; row < weights.rows(); row++)
	{
		for (int col = 0; col < weights.cols(); col++)
		{
			val += pow(weights(row, col), 2) / 2;
		}

		val += pow(weights_b(0, row), 2) / 2;
	}

	return val;
}

void ReorderModel::softmax()
{
	MatrixXd tmpConcat;
	tmpConcat = rae1->RAETree->getRoot()->getVector()->concat(rae2->RAETree->getRoot()->getVector());
	MatrixXd tmpMultiply = tmpConcat * weights.transpose();
	MatrixXd tmpOutput = tmpMultiply + weights_b;


	for(int row = 0;row < outputLayer.rows(); row++)
	{
		for(int col = 0; col < outputLayer.cols(); col++)
		{
			outputLayer(row, col) = tmpOutput(row, col);
		}
	}

	double result;

	result = exp(outputLayer(0, 0))/(exp(outputLayer(0, 0)) + exp(outputLayer(0, 1)));

	outputLayer(0, 0) = result;
	outputLayer(0, 1) = 1-result;	

	softmaxLayer(0, 0) = log(result);
	softmaxLayer(0, 1) = log(1 - result);

	if(!finite(softmaxLayer(0, 0)))
	{
		softmaxLayer(0, 0) = -1 * numeric_limits<double>::max();
	}

	if(!finite(softmaxLayer(0, 1) <= -1 * numeric_limits<double>::max()))
	{       
		softmaxLayer(0, 1) = -1 * numeric_limits<double>::max();
	}
}

void ReorderModel::getData(string bp1, string bp2)
{
	rae1->buildTree(bp1);

	rae2->buildTree(bp2);

	softmax();
}

void ReorderModel::trainRM(MatrixXd y, bool isSoftmax)
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

	MatrixXd theta = MatrixXd(delWeight_b.rows(), delWeight_b.cols());

	//对W和Wb求导
	if(isSoftmax)
	{
		for(int row = 0; row < weights.rows(); row++)
		{
			double result;
			result = (outputLayer(0, row) - y(0, row)) * (exp(outputLayer(0, 0)) * exp(outputLayer(0, 1)) / pow(exp(outputLayer(0, 0)) + exp(outputLayer(0, 1)), 2) );

			for(int col = 0; col < weights.cols(); col++)
			{
				MatrixXd tmpX = rae1->RAETree->getRoot()->getVector()->concat(rae2->RAETree->getRoot()->getVector());
				delWeight(row, col) = delWeight(row, col) + p * result * tmpX(0, col);
			}

			delWeight_b(0, row) = delWeight_b(0, row) + p * result;
			/*	
			cout << softmaxLayer(0, row) << endl;
			cout << y(0, row) << endl;
			cout << outputLayer(0, row) << endl;
			cout << p << " " << result << endl;
			*/
			theta(0, row) = result;
		}

	}
	else
	{
		for(int row = 0; row < weights.rows(); row++)
		{
			double result;
			if(y(0, row) == 1)
			{
				result = (1-outputLayer(0, row));
			}
			else
			{
				result = 0;
			}

			for(int col = 0; col < weights.cols(); col++)
			{
				MatrixXd tmpX = rae1->RAETree->getRoot()->getVector()->concat(rae2->RAETree->getRoot()->getVector());
				delWeight(row, col) = delWeight(row, col) - p * result * tmpX(0, col);
			}

			delWeight_b(0, row) = delWeight_b(0, row) - p * result;
			theta(0, row) = result;
		}
	}

	//对W1和Wb1求导
	Node* preNode1 = rae1->RAETree->getRoot();
	Node* preNode2 = rae2->RAETree->getRoot();

	MatrixXd tmp = theta;
	theta = tmp * weights;

	MatrixXd theta1 = MatrixXd(theta.rows(), theta.cols()/2);
	MatrixXd theta2 = MatrixXd(theta.rows(), theta.cols()/2);

	for(int i = 0; i < theta1.cols(); i++)
	{
		theta1(0, i) = theta(0, i);
		theta2(0, i) = theta(0, i+theta1.cols());
	}

	while(preNode1->getNodeType() != BASED_NODE)
	{
		for(int row = 0; row < vecSize; row++)
		{
			for(int col = 0; col < vecSize*2; col++)
			{
				if(col < vecSize)
				{
					delWeight(row, col) = delWeight(row, col) + p * theta1(0, row) * (preNode1->getLeftChildNode()->getVector())(0, col);
				}
				else
				{
					delWeight(row, col) = delWeight(row, col) + p * theta1(0, row) * (preNode1->getRightChildNode()->getVector())(0, col-vecSize);
				}
			}

			delWeight_b(0, row) = delWeight_b(0, row) + p * theta1(0, row);
		}

		tmp = theta1;
		theta1 = tmp * rae1->weights1;

		MatrixXd tmpTheta = MatrixXd(theta.rows(), theta.cols()/2);

		for(int i = 0; i < theta1.cols(); i++)
		{
			tmpTheta(0, i) = theta1(0, i);
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
					delWeight(row, col) = delWeight(row, col) + p * theta2(0, row) * (preNode2->getLeftChildNode()->getVector())(0, col);
				}
				else
				{
					delWeight(row, col) = delWeight(row, col) + p * theta2(0, row) * (preNode2->getRightChildNode()->getVector())(0, col-vecSize);
				}
			}

			delWeight_b(0, row) = delWeight_b(0, row) + p * theta2(0, row);
		}

		tmp = theta2;
		theta2 = tmp * rae1->weights1;

		MatrixXd tmpTheta = MatrixXd(theta.rows(), theta.cols()/2);

		for(int i = 0; i < theta2.cols(); i++)
		{
			tmpTheta(0, i) = theta2(0, i);
		}

		theta2 = tmpTheta;
		preNode2 = preNode2->getLeftChildNode();
	}
}
