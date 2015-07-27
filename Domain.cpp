#include "Domain.h"

Domain::Domain(Parameter* para, string domainName)
{
	WordVec* srcWords = new WordVec();
	srcWords->readFile(para, domainName+"Source");

	WordVec* tgtWords = new WordVec();
	tgtWords->readFile(para, domainName+"Target");

	dataFile = para->getPara(domainName + "DataFile");
	iterTime = atoi(para->getPara("IterationTime").c_str());

	srcRM = new ReorderModel(para, srcWords);
	tgtRM = new ReorderModel(para, tgtWords);
}

void Domain::upData()
{
	for(int row = 0; row < srcRM->weights->getRow(); row++)
	{
		for(int col = 0; col < srcRM->weights->getCol(); col++)
		{
			srcRM->weights->setValue(row, col, srcRM->weights->getValue(row, col) - RATE * (srcRM->delWeight->getValue(row, col)/trainingData.size() + ZETA * srcRM->weights->getValue(row, col)));
			tgtRM->weights->setValue(row, col, tgtRM->weights->getValue(row, col) - RATE * (tgtRM->delWeight->getValue(row, col)/trainingData.size() + ZETA * tgtRM->weights->getValue(row, col)));
		}
	}

	for(int row = 0; row < srcRM->weights_b->getRow(); row++)
	{
		for(int col = 0; col < srcRM->weights_b->getCol(); col++)
		{
			srcRM->weights_b->setValue(row, col, srcRM->weights_b->getValue(row, col) - RATE * srcRM->delWeight_b->getValue(row, col)/trainingData.size());
			tgtRM->weights_b->setValue(row, col, tgtRM->weights_b->getValue(row, col) - RATE * tgtRM->delWeight_b->getValue(row, col)/trainingData.size());
		}
	}

	for(int row = 0; row < srcRM->rae1->weights1->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights1->getCol(); col++)
		{
			srcRM->rae1->weights1->setValue(row, col, srcRM->rae1->weights1->getValue(row, col) - RATE * (srcRM->rae1->delWeight1->getValue(row, col) + ZETA * srcRM->rae1->weights1->getValue(row, col)));
			tgtRM->rae1->weights1->setValue(row, col, tgtRM->rae1->weights1->getValue(row, col) - RATE * (tgtRM->rae1->delWeight1->getValue(row, col) + ZETA * tgtRM->rae1->weights1->getValue(row, col)));

			srcRM->rae2->weights1->setValue(row, col, srcRM->rae2->weights1->getValue(row, col) - RATE * (srcRM->rae2->delWeight1->getValue(row, col) + ZETA * srcRM->rae2->weights1->getValue(row, col)));
			tgtRM->rae2->weights1->setValue(row, col, tgtRM->rae2->weights1->getValue(row, col) - RATE * (tgtRM->rae2->delWeight1->getValue(row, col) + ZETA * tgtRM->rae2->weights1->getValue(row, col)));
		}
	}

	for(int row = 0; row < srcRM->rae1->weights_b1->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights_b1->getCol(); col++)
		{
			srcRM->rae1->weights_b1->setValue(row, col, srcRM->rae1->weights_b1->getValue(row, col) - RATE * srcRM->rae1->delWeight1_b->getValue(row, col));
			tgtRM->rae1->weights_b1->setValue(row, col, tgtRM->rae1->weights_b1->getValue(row, col) - RATE * tgtRM->rae1->delWeight1_b->getValue(row, col));

			srcRM->rae2->weights_b1->setValue(row, col, srcRM->rae2->weights_b1->getValue(row, col) - RATE * srcRM->rae2->delWeight1_b->getValue(row, col));
			tgtRM->rae2->weights_b1->setValue(row, col, tgtRM->rae2->weights_b1->getValue(row, col) - RATE * tgtRM->rae2->delWeight1_b->getValue(row, col));
		}
	}

	for(int row = 0; row < srcRM->rae1->weights2->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights2->getCol(); col++)
		{
			srcRM->rae1->weights2->setValue(row, col, srcRM->rae1->weights2->getValue(row, col) - RATE * (srcRM->rae1->delWeight2->getValue(row, col) + ZETA * srcRM->rae1->weights2->getValue(row, col)));
			tgtRM->rae1->weights2->setValue(row, col, tgtRM->rae1->weights2->getValue(row, col) - RATE * (tgtRM->rae1->delWeight2->getValue(row, col) + ZETA * tgtRM->rae1->weights2->getValue(row, col)));

			srcRM->rae2->weights2->setValue(row, col, srcRM->rae2->weights2->getValue(row, col) - RATE * (srcRM->rae2->delWeight2->getValue(row, col) + ZETA * srcRM->rae2->weights2->getValue(row, col)));
			tgtRM->rae2->weights2->setValue(row, col, tgtRM->rae2->weights2->getValue(row, col) - RATE * (tgtRM->rae2->delWeight2->getValue(row, col) + ZETA * tgtRM->rae2->weights2->getValue(row, col)));
		}
	}

	for(int row = 0; row < srcRM->rae1->weights_b2->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights_b2->getCol(); col++)
		{
			srcRM->rae1->weights_b2->setValue(row, col, srcRM->rae1->weights_b2->getValue(row, col) - RATE * srcRM->rae1->delWeight2_b->getValue(row, col));
			tgtRM->rae1->weights_b2->setValue(row, col, tgtRM->rae1->weights_b2->getValue(row, col) - RATE * tgtRM->rae1->delWeight2_b->getValue(row, col));

			srcRM->rae2->weights_b2->setValue(row, col, srcRM->rae2->weights_b2->getValue(row, col) - RATE * srcRM->rae2->delWeight2_b->getValue(row, col));
			tgtRM->rae2->weights_b2->setValue(row, col, tgtRM->rae2->weights_b2->getValue(row, col) - RATE * tgtRM->rae2->delWeight2_b->getValue(row, col));
		}
	}
}

//读取单领域训练数据
void Domain::loadTrainingData()
{
	ifstream in(dataFile.c_str(), ios::in);

	string line;
	while(getline(in, line))
	{
		int order;
		map<string, string> m_tmp;
		vector<string> subOfLine = splitBySpace(line);

		if(subOfLine[0] == "mono")
		{
			order = 1;
		}
		else if(subOfLine[0] == "invert")
		{
			order = 0;
		}

		for(int i = 1; i < subOfLine.size(); i++)
		{
			m_tmp.insert(make_pair(subOfLine[i].substr(0, 3), subOfLine[i].substr(4, subOfLine[i].size()-4)));
		}

		trainingData.push_back(make_pair(order, m_tmp));
	}
}

double Domain::loss(int ind)
{
	double lossVal = 0;

	lossVal += ALPHA * srcRM->rae1->loss();
	lossVal += ALPHA * srcRM->rae2->loss();
	lossVal += ALPHA * tgtRM->rae1->loss();
	lossVal += ALPHA * tgtRM->rae2->loss();

	for(int i = 0; i < 2; i++)
	{
		lossVal += GAMMA * pow(srcRM->softmaxLayer->getValue(0, i) - tgtRM->softmaxLayer->getValue(0, i), 2) / 2;
	}

	if(trainingData[ind].first == 1)
	{
		lossVal += BETA * (pow(srcRM->softmaxLayer->getValue(0, 0) - 1, 2) + pow(srcRM->softmaxLayer->getValue(0, 1) - 0, 2))/2;
		lossVal += BETA * (pow(tgtRM->softmaxLayer->getValue(0, 0) - 1, 2) + pow(tgtRM->softmaxLayer->getValue(0, 1) - 0, 2))/2;
	}
	else
	{
		lossVal += BETA * (pow(srcRM->softmaxLayer->getValue(0, 0) - 0, 2) + pow(srcRM->softmaxLayer->getValue(0, 1) - 1, 2))/2;
		lossVal += BETA * (pow(tgtRM->softmaxLayer->getValue(0, 0) - 0, 2) + pow(tgtRM->softmaxLayer->getValue(0, 1) - 1, 2))/2;
	}

	lossVal += ZETA * (srcRM->decay() + tgtRM->decay() + srcRM->rae1->decay() + srcRM->rae2->decay() + tgtRM->rae1->decay() + tgtRM->rae2->decay());

	return lossVal;
}

void Domain::training()
{
	for(int count = 0; count < iterTime; count++)
	{
		cout << "loss value : " << loss(0) << endl;
		//一轮训练
		for(int i = 0; i < trainingData.size(); i++)
		{
			srcRM->getData(trainingData[i].second["ct1"], trainingData[i].second["ct2"]);
			tgtRM->getData(trainingData[i].second["et1"], trainingData[i].second["et2"]);

			srcRM->rae1->trainRecError();
			srcRM->rae2->trainRecError();
			tgtRM->rae1->trainRecError();
			tgtRM->rae2->trainRecError();

			srcRM->trainRM(tgtRM->softmaxLayer, true);
			tgtRM->trainRM(srcRM->softmaxLayer, true);

			Vector* mono = new Vector(1,2);
			Vector* invert = new Vector(1,2);

			for (int i = 0; i < 2; i++)
			{
				mono->setValue(0, i, 1-i);
				invert->setValue(0, i, i);
			}

			if(trainingData[i].first == 1)
			{
				srcRM->trainRM(mono, false);
				tgtRM->trainRM(mono, false);
			}
			else
			{
				srcRM->trainRM(invert, false);
				tgtRM->trainRM(invert, false);
			}
		}

		upData();
	}
}