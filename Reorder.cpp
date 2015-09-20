#include "Reorder.h"

ReorderModel::ReorderModel(Parameter* para, RAE* rae)
{
	vecSize = atoi(para->getPara("WordVecSize").c_str());
	this->para = para;

	this->rae = rae->copy();
	rae1 = rae->copy();
	rae2 = rae->copy();

	weights = MatrixLBFGS(2, vecSize*2);
	weights_b = MatrixLBFGS(1, 2);

	delWeight = MatrixLBFGS(2, vecSize*2);
	delWeight_b = MatrixLBFGS(1, 2);

	delWeight.setZero();
	delWeight_b.setZero();

	outputLayer = MatrixLBFGS(1, 2);
	softmaxLayer = MatrixLBFGS(1, 2);
}

ReorderModel* ReorderModel::copy()
{
	ReorderModel* rm = new ReorderModel(para, rae);

	rm->weights = weights;
	rm->weights_b = weights_b;

	return rm;
}

void ReorderModel::updateWeights(const lbfgsfloatval_t* x)
{
	lbfgsfloatval_t* cX = const_cast<lbfgsfloatval_t*>(x);

	weights = Map<MatrixLBFGS>(cX, 2, vecSize*2);
	weights_b = Map<MatrixLBFGS>(cX + 2*2*vecSize, 1,2);

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

	delWeight.setZero();
	delWeight_b.setZero();
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

void ReorderModel::trainRM(MatrixLBFGS y, int Type)
{
	lbfgsfloatval_t p;
	if(Type == EDIS)
	{
		p = GAMMA;
	}
	else if(Type == EREO)
	{
		p = BETA;
	}

	MatrixLBFGS theta = MatrixLBFGS(delWeight_b.rows(), delWeight_b.cols());

	//对W和Wb求导
	if(Type == EDIS)
	{
		for(int row = 0; row < weights.rows(); row++)
		{
			lbfgsfloatval_t result;
			result = (2*y(0, row)-1) * (exp(outputLayer(0, 0)) * exp(outputLayer(0, 1)));

			for(int col = 0; col < weights.cols(); col++)
			{
				MatrixLBFGS tmpX = concatMatrix(rae1->RAETree->getRoot()->getVector(),rae2->RAETree->getRoot()->getVector());
				delWeight(row, col) = delWeight(row, col) - p * result * tmpX(0, col);
			}

			delWeight_b(0, row) = delWeight_b(0, row) - p * result;


			theta(0, row) = result;
		}

	}
	else if(Type == EREO)
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
			theta(0, row) = p * result;
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

	rae1->recurDel(preNode1, theta1);
	rae2->recurDel(preNode2, theta2);
}

void ReorderModel::trainOnUnlabel(lbfgsfloatval_t ave_p, lbfgsfloatval_t amountOfDomain)
{
	MatrixLBFGS theta = MatrixLBFGS(delWeight_b.rows(), delWeight_b.cols());

	//对W和Wb求导

	for(int row = 0; row < weights.rows(); row++)
	{
		lbfgsfloatval_t result;
		result = (2*ave_p-1) * (exp(outputLayer(0, 0)) * exp(outputLayer(0, 1))) * 4 / amountOfDomain;

		for(int col = 0; col < weights.cols(); col++)
		{
			MatrixLBFGS tmpX = concatMatrix(rae1->RAETree->getRoot()->getVector(),rae2->RAETree->getRoot()->getVector());
			delWeight(row, col) = delWeight(row, col) - DELTA * result * tmpX(0, col);
		}

		delWeight_b(0, row) = delWeight_b(0, row) - DELTA * result;

		theta(0, row) = DELTA * result;
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

	rae1->recurDel(preNode1, theta1);
	rae2->recurDel(preNode2, theta2);
}
