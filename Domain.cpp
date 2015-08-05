#include "Domain.h"

Domain::Domain(Parameter* para, string domainName, WordVec* srcWords, WordVec* tgtWords)
{
	/*
	WordVec* srcWords = new WordVec();
	srcWords->readFile(para, domainName+"Target");

	WordVec* tgtWords = new WordVec();
	tgtWords->readFile(para, domainName+"Source");
	*/
	dataFile = para->getPara(domainName + "DataFile");
	iterTime = atoi(para->getPara("IterationTime").c_str());

	this->domainName = domainName;
	out.open(string("./log/"+ domainName + "/" + domainName+".log").c_str(), ios::out);
	
	srcRM = new ReorderModel(para, srcWords);
	tgtRM = new ReorderModel(para, tgtWords);
}

void Domain::upData()
{
	for(int row = 0; row < srcRM->weights.rows(); row++)
	{
		for(int col = 0; col < srcRM->weights.cols(); col++)
		{
			srcRM->weights(row, col) = srcRM->weights(row, col) - RATE * (srcRM->delWeight(row, col) + ZETA * srcRM->weights(row, col));
			tgtRM->weights(row, col) = tgtRM->weights(row, col) - RATE * (tgtRM->delWeight(row, col) + ZETA * tgtRM->weights(row, col));
		}
	}

	for(int row = 0; row < srcRM->weights_b.rows(); row++)
	{
		for(int col = 0; col < srcRM->weights_b.cols(); col++)
		{
			srcRM->weights_b(row, col) = srcRM->weights_b(row, col) - RATE * srcRM->delWeight_b(row, col);
			tgtRM->weights_b(row, col) = tgtRM->weights_b(row, col) - RATE * tgtRM->delWeight_b(row, col);
		}
	}

	for(int row = 0; row < srcRM->rae1->weights1.rows(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights1.cols(); col++)
		{
			srcRM->rae->weights1(row, col) = srcRM->rae->weights1(row, col) - RATE * (srcRM->rae1->delWeight1(row, col) + ZETA * srcRM->rae->weights1(row, col));
			tgtRM->rae->weights1(row, col) = tgtRM->rae->weights1(row, col) - RATE * (tgtRM->rae1->delWeight1(row, col) + ZETA * tgtRM->rae->weights1(row, col));

			srcRM->rae->weights1(row, col) = srcRM->rae->weights1(row, col) - RATE * (srcRM->rae2->delWeight1(row, col) + ZETA * srcRM->rae->weights1(row, col));
			tgtRM->rae->weights1(row, col) = tgtRM->rae->weights1(row, col) - RATE * (tgtRM->rae2->delWeight1(row, col) + ZETA * tgtRM->rae->weights1(row, col));
		}
	}

	for(int row = 0; row < srcRM->rae1->weights_b1.rows(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights_b1.cols(); col++)
		{
			srcRM->rae->weights_b1(row, col) = srcRM->rae->weights_b1(row, col) - RATE * srcRM->rae1->delWeight1_b(row, col);
			tgtRM->rae->weights_b1(row, col) = tgtRM->rae->weights_b1(row, col) - RATE * tgtRM->rae1->delWeight1_b(row, col);

			srcRM->rae->weights_b1(row, col) = srcRM->rae->weights_b1(row, col) - RATE * srcRM->rae2->delWeight1_b(row, col);
			tgtRM->rae->weights_b1(row, col) = tgtRM->rae->weights_b1(row, col) - RATE * tgtRM->rae2->delWeight1_b(row, col);
		}
	}

	for(int row = 0; row < srcRM->rae1->weights2.rows(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights2.cols(); col++)
		{
			srcRM->rae->weights2(row, col) = srcRM->rae->weights2(row, col) - RATE * (srcRM->rae1->delWeight2(row, col) + ZETA * srcRM->rae->weights2(row, col));
			tgtRM->rae->weights2(row, col) = tgtRM->rae->weights2(row, col) - RATE * (tgtRM->rae1->delWeight2(row, col) + ZETA * tgtRM->rae->weights2(row, col));

			srcRM->rae->weights2(row, col) = srcRM->rae->weights2(row, col) - RATE * (srcRM->rae2->delWeight2(row, col) + ZETA * srcRM->rae->weights2(row, col));
			tgtRM->rae->weights2(row, col) = tgtRM->rae->weights2(row, col) - RATE * (tgtRM->rae2->delWeight2(row, col) + ZETA * tgtRM->rae->weights2(row, col));
		}
	}

	for(int row = 0; row < srcRM->rae1->weights_b2.rows(); row++)
	{
		for(int col = 0; col < srcRM->rae1->weights_b2.cols(); col++)
		{
			srcRM->rae->weights_b2(row, col) = srcRM->rae->weights_b2(row, col) - RATE * srcRM->rae1->delWeight2_b(row, col);
			tgtRM->rae->weights_b2(row, col) = tgtRM->rae->weights_b2(row, col) - RATE * tgtRM->rae1->delWeight2_b(row, col);

			srcRM->rae->weights_b2(row, col) = srcRM->rae->weights_b2(row, col) - RATE * srcRM->rae2->delWeight2_b(row, col);
			tgtRM->rae->weights_b2(row, col) = tgtRM->rae->weights_b2(row, col) - RATE * tgtRM->rae2->delWeight2_b(row, col);
		}
	}

	srcRM->delWeight.setZero();
	srcRM->delWeight_b.setZero();
	tgtRM->delWeight.setZero();
	tgtRM->delWeight_b.setZero();

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

	in.close();
}

//获取单领域的loss value
double Domain::loss(int ind)
{
	double lossVal = 0;

	lossVal += ALPHA * srcRM->rae1->loss();
	lossVal += ALPHA * srcRM->rae2->loss();
	lossVal += ALPHA * tgtRM->rae1->loss();
	lossVal += ALPHA * tgtRM->rae2->loss();
	
	/*
	cout << "srcRM->rae1->loss: " << srcRM->rae1->loss() << endl;
	cout << "srcRM->rae2->loss: " << srcRM->rae2->loss() << endl;
	cout << "tgtRM->rae1->loss: " << tgtRM->rae1->loss() << endl;
	cout << "tgtRM->rae2->loss: " << tgtRM->rae2->loss() << endl;
	*/

	cout << "loss: " << lossVal << endl;

	for(int i = 0; i < 2; i++)
	{
		lossVal += GAMMA * pow(srcRM->outputLayer(0, i) - tgtRM->outputLayer(0, i), 2) / 2;
	}

	cout << "Src softmax: [" << srcRM->softmaxLayer(0, 0) << " , " << srcRM->softmaxLayer(0, 1) << "]" << endl;	
	cout << "Tgt softmax: [" << tgtRM->softmaxLayer(0, 0) << " , " << tgtRM->softmaxLayer(0, 1) << "]" << endl; 	
 	cout << "Src output: [" << srcRM->outputLayer(0, 0) << " , " << srcRM->outputLayer(0, 1) << "]" << endl; 
    cout << "Tgt output: [" << tgtRM->outputLayer(0, 0) << " , " << tgtRM->outputLayer(0, 1) << "]" << endl; 
	
	if(trainingData[ind].first == 1)
	{
		lossVal += BETA * srcRM->softmaxLayer(0, 0) * -1.0;
		lossVal += BETA * tgtRM->softmaxLayer(0, 0) * -1.0;
	}
	else
	{
		lossVal += BETA * srcRM->softmaxLayer(0, 1) * -1.0;
		lossVal += BETA * tgtRM->softmaxLayer(0, 1) * -1.0;
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
		//一轮训练
		//for(int i = trainingData.size()-20000; i < trainingData.size(); i++)
		for(int i = 0; i < trainingData.size(); i++)
		{
			srand((unsigned)time(0));
			int pos = rand()%trainingData.size();

			//获取实例
			srcRM->getData(trainingData[pos].second["ct1"], trainingData[i].second["ct2"]);
			tgtRM->getData(trainingData[pos].second["et1"], trainingData[i].second["et2"]);
			
			if(i >= trainingData.size()-10 && i < trainingData.size())
			{
				out<< count << " : " << pos << "th's " << "loss value : " << loss(pos) << endl;
			}

			//对rae求导
			srcRM->rae1->trainRecError();
			srcRM->rae2->trainRecError();
			tgtRM->rae1->trainRecError();
			tgtRM->rae2->trainRecError();

			//对调序模型求导(Edis)
			srcRM->trainRM(tgtRM->outputLayer, true);
			tgtRM->trainRM(srcRM->outputLayer, true);

			//Ereo
			MatrixXd mono = MatrixXd(1,2);
			MatrixXd invert = MatrixXd(1,2);

			for (int j = 0; j < 2; j++)
			{
				mono(0, j) = 1-j;
				invert(0, j) = j;
			}

			if(trainingData[pos].first == 1)
			{
				srcRM->trainRM(mono, false);
				tgtRM->trainRM(mono, false);
			}
			else
			{
				srcRM->trainRM(invert, false);
				tgtRM->trainRM(invert, false);
			}

			mono.resize(0,0);
			invert.resize(0,0);
			
			//更新权重
			upData();
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
	srcWLog << srcRM->weights << endl;
	tgtWLog << tgtRM->weights << endl;

	srcWLog << "b: \n";
	tgtWLog << "b: \n";
	srcWLog << srcRM->weights_b << endl;
	tgtWLog << tgtRM->weights_b << endl;

	srcWLog << "RAE: \nW1: \n";
	tgtWLog << "RAE: \nW1: \n";
	srcWLog << srcRM->rae->weights1 << endl;
	tgtWLog << tgtRM->rae->weights1 << endl;

	srcWLog << "b1: \n";
	tgtWLog << "b1: \n";
	srcWLog << srcRM->rae->weights_b1 << endl;
	tgtWLog << tgtRM->rae->weights_b1 << endl;

	srcWLog << "W2: \n";
	tgtWLog << "W2: \n";
	srcWLog << srcRM->rae->weights2 << endl;
	tgtWLog << tgtRM->rae->weights2 << endl;

	srcWLog << "b2: \n";
	tgtWLog << "b2: \n";
	srcWLog << srcRM->rae->weights_b2 << endl;
	tgtWLog << tgtRM->rae->weights_b2 << endl;

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

	//for(int i = trainingData.size()-20000; i < trainingData.size(); i++)
	for(int i = 0; i < trainingData.size(); i++)
	{
		srcRM->getData(trainingData[i].second["ct1"], trainingData[i].second["ct2"]);
		tgtRM->getData(trainingData[i].second["et1"], trainingData[i].second["et2"]);

		if(srcRM->outputLayer(0, 0) > srcRM->outputLayer(0,1))
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

		if(tgtRM->outputLayer(0, 0) > tgtRM->outputLayer(0,1))
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

		srcOut << trainingData[i].first << "\t\t[" << srcRM->outputLayer(0,0) << " , " << srcRM->outputLayer(0,1) << "]" << endl;
		tgtOut << trainingData[i].first << "\t\t[" << tgtRM->outputLayer(0,0) << " , " << tgtRM->outputLayer(0,1) << "]" << endl;
	}

	srcOut << "Precision: " << (double)srcCount/trainingData.size() << endl;
	tgtOut << "Precision: " << (double)tgtCount/trainingData.size() << endl;
	//srcOut << "Precision: " << (double)srcCount/20000 << endl;
        //tgtOut << "Precision: " << (double)tgtCount/20000 << endl;

	srcOut.close();
	tgtOut.close();
}
