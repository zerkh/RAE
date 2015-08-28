#include "Reorder.h"

ReorderModel::ReorderModel(Parameter* para, WordVec* words)
{
	vecSize = atoi(para->getPara("WordVecSize").c_str());

	rae = new RAE(para, words);
	rae1 = NULL;
	rae2 = NULL;

	delWeight = MatrixLBFGS(weights.rows(), weights.cols());
	delWeight_b = MatrixLBFGS(weights_b.rows(), weights_b.cols());

	delWeight.setZero();
	delWeight_b.setZero();

	outputLayer = MatrixLBFGS(1, 2);
	softmaxLayer = MatrixLBFGS(1, 2);
}

void ReorderModel::updateWeights(const lbfgsfloatval_t* x, int base)
{
	lbfgsfloatval_t* cX = const_cast<lbfgsfloatval_t*>(x);

	weights = Map<MatrixLBFGS>(cX+base, 2, vecSize*2);
	weights_b = Map<MatrixLBFGS>(cX+base+2*2*vecSize, 1,2);

	rae->weights1 = Map<MatrixLBFGS>(cX+base+getRMWeightSize(), vecSize, 2*vecSize);
	rae->weights_b1 = Map<MatrixLBFGS>(cX+base+getRMWeightSize()+2*vecSize*vecSize, 1, vecSize);
	rae->weights2 = Map<MatrixLBFGS>(cX+base+getRMWeightSize()+2*vecSize*vecSize+vecSize, 2*vecSize, vecSize);
	rae->weights_b2 = Map<MatrixLBFGS>(cX+base+getRMWeightSize()+2*vecSize*vecSize+vecSize+2*vecSize*vecSize, 1, 2*vecSize);

	if(rae1)
	{
		delete rae1;
		rae1 = NULL;
	}

	if(rae2)
	{
		delete rae2;
		rae2 = NULL;
	}

	rae1 = rae->copy();
	rae2 = rae->copy();
}

lbfgsfloatval_t ReorderModel::decay()
{
	lbfgsfloatval_t val = 0;

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

int ReorderModel::getRMWeightSize()
{
	return (vecSize*2*2 + 2);
}

void ReorderModel::softmax()
{
	MatrixLBFGS tmpConcat;
	tmpConcat = concatMatrix(rae1->RAETree->getRoot()->getVector(),rae2->RAETree->getRoot()->getVector());
	MatrixLBFGS tmpMultiply = tmpConcat * weights.transpose();
	MatrixLBFGS tmpOutput = tmpMultiply + weights_b;

	for(int row = 0;row < outputLayer.rows(); row++)
	{
		for(int col = 0; col < outputLayer.cols(); col++)
		{
			outputLayer(row, col) = tmpOutput(row, col);
		}
	}

	lbfgsfloatval_t result, o1, o2;

	o1 = exp(outputLayer(0,0));
	o2 = exp(outputLayer(0,1));
	if(exp(outputLayer(0,1)) == 0 && exp(outputLayer(0,0)) == 0)
	{
		o1 = 1;
		o2 = 1;
	}
	else
	{
		if(!finite(exp(outputLayer(0,1))) )
		{
			o2 = numeric_limits<lbfgsfloatval_t>::max() * 0.1;
		}
		if(!finite(exp(outputLayer(0,0))) )
		{
			o1 = numeric_limits<lbfgsfloatval_t>::max() * 0.1;
		}
	}
	result = o1/(o1+o2);	

	outputLayer(0, 0) = result;
	outputLayer(0, 1) = 1-result;
	softmaxLayer(0, 0) = log(result);
	softmaxLayer(0, 1) = log(1 - result);

	if(!finite(softmaxLayer(0, 0)))
	{
		softmaxLayer(0, 0) = -1 * numeric_limits<lbfgsfloatval_t>::max();
	}

	if(!finite(softmaxLayer(0, 1)))
	{       
		softmaxLayer(0, 1) = -1 * numeric_limits<lbfgsfloatval_t>::max();
	}
}

void ReorderModel::getData(string bp1, string bp2)
{
	rae1->buildTree(bp1);

	rae2->buildTree(bp2);

	softmax();
}

void ReorderModel::trainRM(MatrixLBFGS y, bool isSoftmax)
{
	lbfgsfloatval_t p;
	if(isSoftmax)
	{
		p = GAMMA;
	}
	else
	{
		p = BETA;
	}

	MatrixLBFGS theta = MatrixLBFGS(delWeight_b.rows(), delWeight_b.cols());

	//对W和Wb求导
	if(isSoftmax)
	{
		for(int row = 0; row < weights.rows(); row++)
		{
			lbfgsfloatval_t result;
			result = (outputLayer(0, row) - y(0, row)) * (exp(outputLayer(0, 0)) * exp(outputLayer(0, 1)));

			for(int col = 0; col < weights.cols(); col++)
			{
				MatrixLBFGS tmpX = concatMatrix(rae1->RAETree->getRoot()->getVector(),rae2->RAETree->getRoot()->getVector());
				delWeight(row, col) = delWeight(row, col) + p * result * tmpX(0, col);
			}

			delWeight_b(0, row) = delWeight_b(0, row) + p * result;
				
					
			theta(0, row) = result;
		}

	}
	else
	{
		for(int row = 0; row < weights.rows(); row++)
		{
			lbfgsfloatval_t result;
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
				MatrixLBFGS tmpX = concatMatrix(rae1->RAETree->getRoot()->getVector(),rae2->RAETree->getRoot()->getVector());
				delWeight(row, col) = delWeight(row, col) - p * result * tmpX(0, col);
			}

			delWeight_b(0, row) = delWeight_b(0, row) - p * result;
			theta(0, row) = result;
		}
	}


	//对W1和Wb1求导
	Node* preNode1 = rae1->RAETree->getRoot();
	Node* preNode2 = rae2->RAETree->getRoot();

	theta = theta * weights;

	MatrixLBFGS theta1 = MatrixLBFGS(theta.rows(), theta.cols()/2);
	MatrixLBFGS theta2 = MatrixLBFGS(theta.rows(), theta.cols()/2);

	for(int i = 0; i < theta1.cols(); i++)
	{
		theta1(0, i) = theta(0, i);
		theta2(0, i) = theta(0, i+theta1.cols());
	}

	while(preNode1->getNodeType() != BASED_NODE)
	{
		for(int col = 0; col < theta1.cols(); col++)
		{
			theta1(0, col) *= (1-pow(preNode1->getVector()(0,col), 2));
		}

		for(int row = 0; row < vecSize; row++)
		{
			for(int col = 0; col < vecSize*2; col++)
			{
				if(col < vecSize)
				{
					rae1->delWeight1(row, col) = rae1->delWeight1(row, col) + p * theta1(0, row) * (preNode1->getLeftChildNode()->getVector())(0, col);
				}
				else
				{
					rae1->delWeight1(row, col) = rae1->delWeight1(row, col) + p * theta1(0, row) * (preNode1->getRightChildNode()->getVector())(0, col-vecSize);
				}
			}

			rae1->delWeight1_b(0, row) = rae1->delWeight1_b(0, row) + p * theta1(0, row);
		}

		theta1 = theta1 * rae1->weights1;

		MatrixLBFGS tmpTheta = MatrixLBFGS(theta.rows(), theta.cols()/2);

		for(int i = 0; i < theta.cols()/2; i++)
		{
			tmpTheta(0, i) = theta1(0, i);
		}

		theta1 = tmpTheta;
		preNode1 = preNode1->getLeftChildNode();
	}

	while(preNode2->getNodeType() != BASED_NODE)
	{
		for(int col = 0; col < theta2.cols(); col++)
		{
			theta2(0, col) *= (1-pow(preNode2->getVector()(0,col), 2));
		}

		for(int row = 0; row < vecSize; row++)
		{
			for(int col = 0; col < vecSize*2; col++)
			{
				if(col < vecSize)
				{
					rae2->delWeight1(row, col) = rae2->delWeight1(row, col) + p * theta2(0, row) * (preNode2->getLeftChildNode()->getVector())(0, col);
				}
				else
				{
					rae2->delWeight1(row, col) = rae2->delWeight1(row, col) + p * theta2(0, row) * (preNode2->getRightChildNode()->getVector())(0, col-vecSize);
				}
			}

			rae2->delWeight2_b(0, row) = rae2->delWeight2_b(0, row) + p * theta2(0, row);
		}

		theta2 = theta2 * rae2->weights1;

		MatrixLBFGS tmpTheta = MatrixLBFGS(theta.rows(), theta.cols()/2);

		for(int i = 0; i < theta.cols()/2; i++)
		{
			tmpTheta(0, i) = theta2(0, i);
		}

		theta2 = tmpTheta;
		preNode2 = preNode2->getLeftChildNode();
	}
}
