#include "WordVec.h"
#include <fstream>
#include <sstream>

map<string, Vector*> WordVec::m_words;
map<string, Vector*> WordVec::m_strings;

WordVec::WordVec()
{
	cout << "Reading word vector..." << endl << endl;

	amountOfWords = 0;
	amountOfStrings = 0;
}

//读取词向量
void WordVec::readFile(Parameter* para)
{
	string filename = para->getPara("WordVecSrcFile");
	int size = atoi(para->getPara("WordVecSize").c_str());

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

		Vector* tmpVec = new Vector(1, size);

		for(int i = 0; i < size; i++)
		{
			tmpVec->setValue(0, i, wordvec[i]);
		}

		this->m_words.insert(make_pair(word, tmpVec));
	}
}


//显示所有词及向量
void WordVec::showWords()
{
	for(map<string, Vector*>::iterator m_it = m_words.begin(); m_it != m_words.end(); m_it++)
	{
		cout << m_it->first << endl;
		m_it->second->showVector();
	}
}

//显示所有词组及向量
void WordVec::showStrings()
{
	for(map<string, Vector*>::iterator m_it = m_strings.begin(); m_it != m_strings.end(); m_it++)
	{
		cout << m_it->first << endl;
		m_it->second->showVector();
	}
}
