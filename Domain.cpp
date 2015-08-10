#include "Domain.h"

Domain::Domain(Parameter* para, string domainName, WordVec* srcWords, WordVec* tgtWords)
{
	/*
	WordVec* srcWords = new WordVec();
	srcWords->readFile(para, domainName+"Target");

	WordVec* tgtWords = new WordVec();
	tgtWords->readFile(para, domainName+"Source");
	*/
	bool isDev = atoi(para->getPara("IsDev").c_str());
	bool isTrain = atoi(para->getPara("IsTrain").c_str());
	bool isTest = atoi(para->getPara("IsTest").c_str());

	if(isDev)
	{
		dataFile = para->getPara(domainName + "DevDataFile");
	}
	else if(isTrain)
	{
		dataFile = para->getPara(domainName + "DataFile");
	}
	else if(isTest)
	{
		dataFile = para->getPara(domainName + "TestDataFile");
	}

	iterTime = atoi(para->getPara("IterationTime").c_str());

	this->domainName = domainName;
	out.open(string("./log/"+ domainName + "/" + domainName+".log").c_str(), ios::out);
	
	srcRM = new ReorderModel(para, srcWords);
	tgtRM = new ReorderModel(para, tgtWords);
}

void Domain::upData()
{
	for(int row = 0; row < srcRM->weights->getRow(); row++)
	{
		for(int col = 0; col < srcRM->weights->getCol(); col++)
		{
			srcRM->weights->setValue(row, col, srcRM->weights->getValue(row, col) - RATE * (srcRM->delWeight->getValue(row, col) + ZETA * srcRM->weights->getValue(row, col)));
			tgtRM->weights->setValue(row, col, tgtRM->weights->getValue(row, col) - RATE * (tgtRM->delWeight->getValue(row, col) + ZETA * tgtRM->weights->getValue(row, col)));
		}
	}

	for(int row = 0; row < srcRM->weights_b->getRow(); row++)
	{
		for(int col = 0; col < srcRM->weights_b->getCol(); col++)
		{
			srcRM->weights_b->setValue(row, col, srcRM->weights_b->getValue(row, col) - RATE * srcRM->delWeight_b->getValue(row, col));
			tgtRM->weights_b->setValue(row, col, tgtRM->weights_b->getValue(row, col) - RATE * tgtRM->delWeight_b->getValue(row, col));
		}
	}

	for(int row = 0; row < srcRM->rae1->weights1->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights1->getCol(); col++)
		{
			srcRM->rae->weights1->setValue(row, col, srcRM->rae->weights1->getValue(row, col) - RATE * (srcRM->rae1->delWeight1->getValue(row, col) + ZETA * srcRM->rae->weights1->getValue(row, col)));
			tgtRM->rae->weights1->setValue(row, col, tgtRM->rae->weights1->getValue(row, col) - RATE * (tgtRM->rae1->delWeight1->getValue(row, col) + ZETA * tgtRM->rae->weights1->getValue(row, col)));

			srcRM->rae->weights1->setValue(row, col, srcRM->rae->weights1->getValue(row, col) - RATE * (srcRM->rae2->delWeight1->getValue(row, col) + ZETA * srcRM->rae->weights1->getValue(row, col)));
			tgtRM->rae->weights1->setValue(row, col, tgtRM->rae->weights1->getValue(row, col) - RATE * (tgtRM->rae2->delWeight1->getValue(row, col) + ZETA * tgtRM->rae->weights1->getValue(row, col)));
		}
	}

	for(int row = 0; row < srcRM->rae1->weights_b1->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights_b1->getCol(); col++)
		{
			srcRM->rae->weights_b1->setValue(row, col, srcRM->rae->weights_b1->getValue(row, col) - RATE * srcRM->rae1->delWeight1_b->getValue(row, col));
			tgtRM->rae->weights_b1->setValue(row, col, tgtRM->rae->weights_b1->getValue(row, col) - RATE * tgtRM->rae1->delWeight1_b->getValue(row, col));

			srcRM->rae->weights_b1->setValue(row, col, srcRM->rae->weights_b1->getValue(row, col) - RATE * srcRM->rae2->delWeight1_b->getValue(row, col));
			tgtRM->rae->weights_b1->setValue(row, col, tgtRM->rae->weights_b1->getValue(row, col) - RATE * tgtRM->rae2->delWeight1_b->getValue(row, col));
		}
	}

	for(int row = 0; row < srcRM->rae1->weights2->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights2->getCol(); col++)
		{
			srcRM->rae->weights2->setValue(row, col, srcRM->rae->weights2->getValue(row, col) - RATE * (srcRM->rae1->delWeight2->getValue(row, col) + ZETA * srcRM->rae->weights2->getValue(row, col)));
			tgtRM->rae->weights2->setValue(row, col, tgtRM->rae->weights2->getValue(row, col) - RATE * (tgtRM->rae1->delWeight2->getValue(row, col) + ZETA * tgtRM->rae->weights2->getValue(row, col)));

			srcRM->rae->weights2->setValue(row, col, srcRM->rae->weights2->getValue(row, col) - RATE * (srcRM->rae2->delWeight2->getValue(row, col) + ZETA * srcRM->rae->weights2->getValue(row, col)));
			tgtRM->rae->weights2->setValue(row, col, tgtRM->rae->weights2->getValue(row, col) - RATE * (tgtRM->rae2->delWeight2->getValue(row, col) + ZETA * tgtRM->rae->weights2->getValue(row, col)));
		}
	}

	for(int row = 0; row < srcRM->rae1->weights_b2->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights_b2->getCol(); col++)
		{
			srcRM->rae->weights_b2->setValue(row, col, srcRM->rae->weights_b2->getValue(row, col) - RATE * srcRM->rae1->delWeight2_b->getValue(row, col));
			tgtRM->rae->weights_b2->setValue(row, col, tgtRM->rae->weights_b2->getValue(row, col) - RATE * tgtRM->rae1->delWeight2_b->getValue(row, col));

			srcRM->rae->weights_b2->setValue(row, col, srcRM->rae->weights_b2->getValue(row, col) - RATE * srcRM->rae2->delWeight2_b->getValue(row, col));
			tgtRM->rae->weights_b2->setValue(row, col, tgtRM->rae->weights_b2->getValue(row, col) - RATE * tgtRM->rae2->delWeight2_b->getValue(row, col));
		}
	}

	srcRM->delWeight->setToZeros();
	srcRM->delWeight_b->setToZeros();
	tgtRM->delWeight->setToZeros();
	tgtRM->delWeight_b->setToZeros();

	delete srcRM->rae1;
	delete srcRM->rae2;

	delete tgtRM->rae1;
	delete tgtRM->rae2;

	srcRM->rae1 = srcRM->rae->copy();
	srcRM->rae2 = srcRM->rae->copy();

	tgtRM->rae1 = tgtRM->rae->copy();
	tgtRM->rae2 = tgtRM->rae->copy();
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
		vector<string> subOfLine = splitBySign(line);

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
			int pos = subOfLine[i].find('=');
			m_tmp.insert(make_pair(subOfLine[i].substr(0, pos), subOfLine[i].substr(pos+1, subOfLine[i].size()-pos)));
		}

		trainingData.push_back(make_pair(order, m_tmp));
	}

	in.close();
}

//获取单领域的loss value
double Domain::loss(int ind)
{
	double lossVal = 0;
	srcRM->getData(trainingData[ind].second["ct1"], trainingData[ind].second["ct2"]);
	tgtRM->getData(trainingData[ind].second["et1"], trainingData[ind].second["et2"]);

	lossVal += ALPHA * srcRM->rae1->loss();
	lossVal += ALPHA * srcRM->rae2->loss();
	lossVal += ALPHA * tgtRM->rae1->loss();
	lossVal += ALPHA * tgtRM->rae2->loss();
	
	cout << "srcRM->rae1->loss: " << srcRM->rae1->loss() << endl;
	cout << "srcRM->rae2->loss: " << srcRM->rae2->loss() << endl;
	cout << "tgtRM->rae1->loss: " << tgtRM->rae1->loss() << endl;
	cout << "tgtRM->rae2->loss: " << tgtRM->rae2->loss() << endl;

	cout << "loss: " << lossVal << endl;

	for(int i = 0; i < 2; i++)
	{
		lossVal += GAMMA * pow(srcRM->softmaxLayer->getValue(0, i) - tgtRM->softmaxLayer->getValue(0, i), 2) / 2;
	}

	cout << "Src softmax: [" << srcRM->softmaxLayer->getValue(0, 0) << " , " << srcRM->softmaxLayer->getValue(0, 1) << "]" << endl;	
	cout << "Tgt softmax: [" << tgtRM->softmaxLayer->getValue(0, 0) << " , " << tgtRM->softmaxLayer->getValue(0, 1) << "]" << endl; 	
 	cout << "Src output: [" << srcRM->outputLayer->getValue(0, 0) << " , " << srcRM->outputLayer->getValue(0, 1) << "]" << endl; 
    cout << "Tgt output: [" << tgtRM->outputLayer->getValue(0, 0) << " , " << tgtRM->outputLayer->getValue(0, 1) << "]" << endl; 
	
	if(trainingData[ind].first == 1)
	{
		lossVal += BETA * srcRM->softmaxLayer->getValue(0, 0) * -1.0;
		lossVal += BETA * tgtRM->softmaxLayer->getValue(0, 0) * -1.0;
	}
	else
	{
		lossVal += BETA * srcRM->softmaxLayer->getValue(0, 1) * -1.0;
		lossVal += BETA * tgtRM->softmaxLayer->getValue(0, 1) * -1.0;
	}

	cout << "After Ereo loss: " << lossVal << endl;	

	lossVal += ZETA * (srcRM->decay() + tgtRM->decay() + srcRM->rae->decay() + tgtRM->rae->decay());

	cout << "After Decay loss: " << lossVal << endl;

	return lossVal;
}

void Domain::training()
{
	for(int count = 0; count < iterTime; count++)
	{
		srand((unsigned)time(0));
		RATE = RATE/2;
		
		cout << "learning rate: " << RATE << endl << endl;

		//一轮训练
		//for(int i = trainingData.size()-100; i < trainingData.size(); i++)
		for(int i = 0; i < trainingData.size(); i++)
		{
			//输出loss
			if(i % 10000 == 0)
			{
				out << count  << ": " << "0th: " << loss(0) << endl;
			}

			int pos = rand()%trainingData.size();

			cout << "Domain 229" << endl;
			//获取实例
			srcRM->getData(trainingData[pos].second["ct1"], trainingData[pos].second["ct2"]);
			tgtRM->getData(trainingData[pos].second["et1"], trainingData[pos].second["et2"]);
			cout << "Domain 233" << endl;

			cout << "Domain 235" << endl;
			//对rae求导
			srcRM->rae1->trainRecError();
			srcRM->rae2->trainRecError();
			tgtRM->rae1->trainRecError();
			tgtRM->rae2->trainRecError();
			cout << "Domain 241" << endl;

			cout << "Domain 243" << endl;
			//对调序模型求导(Edis)
			srcRM->trainRM(tgtRM->softmaxLayer, true);
			tgtRM->trainRM(srcRM->softmaxLayer, true);
			cout << "Domain 247" << endl;

			//Ereo
			Vector* mono = new Vector(1,2);
			Vector* invert = new Vector(1,2);

			for (int j = 0; j < 2; j++)
			{
				mono->setValue(0, j, 1-j);
				invert->setValue(0, j, j);
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
			
			delete mono;
			delete invert;

			cout << "Domain 273" << endl;
			//更新权重
			upData();
			cout << "Domain 276" << endl;
		}
	}

	//记录权重
	logWeights();
}

void Domain::logWeights()
{	
	srcWLog.open(string("./log/"+ domainName + "/" + "src"+domainName+"Weights.log").c_str(), ios::out);
	tgtWLog.open(string("./log/"+ domainName + "/" + "tgt"+domainName+"Weights.log").c_str(), ios::out);

	srcWLog << "RM: \nW: \n";
	tgtWLog << "RM: \nW: \n";

	for(int row = 0; row < srcRM->weights->getRow(); row++)
	{
		for(int col = 0; col < srcRM->weights->getCol(); col++)
		{
			srcWLog << srcRM->weights->getValue(row, col) << " ";
			tgtWLog << tgtRM->weights->getValue(row, col) << " ";
		}
		srcWLog << endl;
		tgtWLog << endl;
	}

	srcWLog << "b: \n";
	tgtWLog << "b: \n";

	for(int row = 0; row < srcRM->weights_b->getRow(); row++)
	{
		for(int col = 0; col < srcRM->weights_b->getCol(); col++)
		{
			srcWLog << srcRM->weights_b->getValue(row, col) << " ";
			tgtWLog << tgtRM->weights_b->getValue(row, col) << " ";
		}
		srcWLog << endl;
		tgtWLog << endl;
	}

	srcWLog << "RAE: \nW1: \n";
	tgtWLog << "RAE: \nW1: \n";

	for(int row = 0; row < srcRM->rae->weights1->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae->weights1->getCol(); col++)
		{
			srcWLog << srcRM->rae->weights1->getValue(row, col) << " ";
			tgtWLog << tgtRM->rae->weights1->getValue(row, col) << " ";
		}
		srcWLog << endl;
		tgtWLog << endl;
	}

	srcWLog << "b1: \n";
	tgtWLog << "b1: \n";
	for(int row = 0; row < srcRM->rae->weights_b1->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae->weights_b1->getCol(); col++)
		{
			srcWLog << srcRM->rae->weights_b1->getValue(row, col) << " ";
			tgtWLog << tgtRM->rae->weights_b1->getValue(row, col) << " ";
		}
		srcWLog << endl;
		tgtWLog << endl;
	}

	srcWLog << "W2: \n";
	tgtWLog << "W2: \n";
	for(int row = 0; row < srcRM->rae->weights2->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae->weights2->getCol(); col++)
		{
			srcWLog << srcRM->rae->weights2->getValue(row, col) << " ";
			tgtWLog << tgtRM->rae->weights2->getValue(row, col) << " ";
		}
		srcWLog << endl;
		tgtWLog << endl;
	}

	srcWLog << "b2: \n";
	tgtWLog << "b2: \n";
	for(int row = 0; row < srcRM->rae->weights_b2->getRow(); row++)
	{
		for(int col = 0; col < srcRM->rae->weights_b2->getCol(); col++)
		{
			srcWLog << srcRM->rae->weights_b2->getValue(row, col) << " ";
			tgtWLog << tgtRM->rae->weights_b2->getValue(row, col) << " ";
		}
		srcWLog << endl;
		tgtWLog << endl;
	}

	srcWLog.close();
	srcWLog.close();
}

void Domain::test()
{
	srcOut.open(string("./log/"+ domainName + "/" + "src"+domainName+".log").c_str(), ios::out);
	tgtOut.open(string("./log/"+ domainName + "/" + "tgt"+domainName+".log").c_str(), ios::out);

	srcOut << "True value\t\tPredict value" << endl;
	tgtOut << "True value\t\tPredict value" << endl;

	int srcCount = 0;
	int tgtCount = 0;

	//for(int i = 0; i < 200; i++)
	for(int i = 0; i < trainingData.size(); i++)
	{
		srcRM->getData(trainingData[i].second["ct1"], trainingData[i].second["ct2"]);
		tgtRM->getData(trainingData[i].second["et1"], trainingData[i].second["et2"]);

		if(srcRM->outputLayer->getValue(0, 0) > srcRM->outputLayer->getValue(0,1))
		{
			if(trainingData[i].first == 1)
			{
				srcCount++;
			}
		}
		else
		{
			if(trainingData[i].first == 0)
			{
				srcCount++;
			}
		}

		if(tgtRM->outputLayer->getValue(0, 0) > tgtRM->outputLayer->getValue(0,1))
		{
			if(trainingData[i].first == 1)
			{
				tgtCount++;
			}
		}
		else
                {
                        if(trainingData[i].first == 0)
                        {
                                tgtCount++;
                        }
                }

		srcOut << trainingData[i].first << "\t\t[" << srcRM->outputLayer->getValue(0,0) << " , " << srcRM->outputLayer->getValue(0,1) << "]" << endl;
		tgtOut << trainingData[i].first << "\t\t[" << tgtRM->outputLayer->getValue(0,0) << " , " << tgtRM->outputLayer->getValue(0,1) << "]" << endl;
	}

	srcOut << "Precision: " << (double)srcCount/trainingData.size() << endl;
	tgtOut << "Precision: " << (double)tgtCount/trainingData.size() << endl;
	//srcOut << "Precision: " << (double)srcCount/200 << endl;
	//tgtOut << "Precision: " << (double)tgtCount/200 << endl;

	srcOut.close();
	tgtOut.close();
}
