#include "RAE.h"

RAE::RAE(Parameter* para, WordVec* words)
{
	vecSize = atoi(para->getPara("WordVecSize").c_str());
	this->words = words;

	weights1 = new Vector(vecSize, vecSize*2);
	weights2 = new Vector(vecSize*2, vecSize);
	weights_b1 = new Vector(1, vecSize);
	weights_b2 = new Vector(1, vecSize*2);

	weights1->randInitVector();
	weights2->randInitVector();
	weights_b1->randInitVector();
	weights_b2->randInitVector();
}

//显示参数
void RAE::showWeights()
{
	cout << "W1\t" << "Row: " << weights1->getRow() << "\tCol: " << weights1->getCol() << endl;
	for(int row = 0; row < weights1->getRow(); row++)
	{
		for(int col = 0; col < weights1->getCol(); col++)
		{
			cout << weights1->getValue(row, col) << " ";
		}

		cout << endl;
	}

	cout << "B1\t" << "Row: " << weights_b1->getRow() << "\tCol: " << weights_b1->getCol() << endl;
	for(int row = 0; row < weights_b1->getRow(); row++)
	{
		for(int col = 0; col < weights_b1->getCol(); col++)
		{
			cout << weights_b1->getValue(row, col) << " ";
		}

		cout << endl;
	}

	cout << "W2\t" << "Row: " << weights2->getRow() << "\tCol: " << weights2->getCol() << endl;
	for(int row = 0; row < weights2->getRow(); row++)
	{
		for(int col = 0; col < weights2->getCol(); col++)
		{
			cout << weights2->getValue(row, col) << " ";
		}

		cout << endl;
	}

	cout << "B2\t" << "Row: " << weights_b2->getRow() << "\tCol: " << weights_b2->getCol() << endl;
	for(int row = 0; row < weights_b2->getRow(); row++)
	{
		for(int col = 0; col < weights_b2->getCol(); col++)
		{
			cout << weights_b2->getValue(row, col) << " ";
		}

		cout << endl;
	}
}

//保存参数到文件
void RAE::logWeights(Parameter* para)
{
	string filename = para->getPara("RAEWeightsLogFile");

	ofstream out(filename.c_str(), ios::out);

	out << "W1" << endl;
	for(int row = 0; row < weights1->getRow(); row++)
	{
		for(int col = 0; col < weights1->getCol(); col++)
		{
			out << weights1->getValue(row, col) << " ";
		}

		out << endl;
	}

	out << "B1" << endl;
	for(int row = 0; row < weights_b1->getRow(); row++)
	{
		for(int col = 0; col < weights_b1->getCol(); col++)
		{
			out << weights_b1->getValue(row, col) << " ";
		}

		out << endl;
	}

	out << "W2" << endl;
	for(int row = 0; row < weights2->getRow(); row++)
	{
		for(int col = 0; col < weights2->getCol(); col++)
		{
			out << weights2->getValue(row, col) << " ";
		}

		out << endl;
	}

	out << "B2" << endl;
	for(int row = 0; row < weights_b2->getRow(); row++)
	{
		for(int col = 0; col < weights_b2->getCol(); col++)
		{
			out << weights_b2->getValue(row, col) << " ";
		}

		out << endl;
	}
}

//通过RAE求短语向量
p_StringVec RAE::getStringVec(string word1 ,string word2)
{
	Vector* wordVec1 = words->m_words[word1];

	Vector* wordVec2 = words->m_words[word2];

	Vector* inputVec = wordVec1->concat(wordVec2);

	Vector* outputVec = inputVec->multiply(weights1, true)->add(weights_b1);

	return make_pair(word1 + " " + word2, outputVec);
}

//训练RAE
void RAE::trainRAE()
{

}

int RAE::getVecSize()
{
	return vecSize;
}

double RAE::loss(Vector* inputLayer, Vector* recLayer)
{
	double lossVal = 0;

	for(int col = 0; col < inputLayer->getCol(); col++)
	{
		lossVal += pow(inputLayer->getValue(0, col)-recLayer->getValue(0, col), 2);
	}

	return lossVal/2;
}