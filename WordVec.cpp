#include "WordVec.h"

map<string, MatrixXd> WordVec::m_words;
map<string, MatrixXd> WordVec::m_strings;

WordVec::WordVec()
{
	cout << "Reading word vector..." << endl << endl;

	amountOfWords = 0;
	amountOfStrings = 0;
}

//读取词向量
void WordVec::readFile(Parameter* para, string titleStr)
{
	string filename = para->getPara(titleStr+"WordVecFile");
	int size = atoi(para->getPara("WordVecSize").c_str());

	cout << "Reading " << filename << "......" << endl << endl;

	ifstream fin(filename.c_str(), ios::in);

	if(!fin)
	{
		cerr << "Open " + filename << " fail!" << endl;
		exit(-1);
	}

	string line;

	getline(fin, line);
	stringstream strin(line);
	strin >> this->amountOfWords;
	strin >> size;

	while(getline(fin, line))
	{
		stringstream strin(line);

		string word;
		double* wordvec = new double[size];

		strin >> word;
		for(int i = 0; i < size; i++)
		{
			strin >> wordvec[i];
		}

		MatrixXd tmpVec = MatrixXd(1, size);

		for(int i = 0; i < size; i++)
		{
			tmpVec(0, i) = wordvec[i];
		}

		this->m_words.insert(make_pair(word, tmpVec));
	}

	cout << "Finish reading" << filename << endl << endl;
}

bool WordVec::isInDict(string word)
{
	if (m_words.find(word) != m_words.end())
	{
		return true;
	}
	else
	{
		return false;
	}
}

//显示所有词及向量
void WordVec::showWords()
{
	for(map<string, MatrixXd>::iterator m_it = m_words.begin(); m_it != m_words.end(); m_it++)
	{
		cout << m_it->first << endl;
		cout << m_it->second << endl;
	}
}

//显示所有词组及向量
void WordVec::showStrings()
{
	for(map<string, MatrixXd>::iterator m_it = m_strings.begin(); m_it != m_strings.end(); m_it++)
	{
		cout << m_it->first << endl;
		cout << m_it->second << endl;
	}
}
