#ifndef WORDVEC_H
#define WORDVEC_H

#include <iostream>
#include <map>
#include <cstdlib>

using namespace std;

class WordVec
{
private:
	typedef pair<string, double*>	Word;
	string							filename;
	int								size;				//��������ά��
	int								amountOfWords;		//�ʻ���

public:
	map<string, double*>			m_words;

public:
	WordVec(string filename);

	void readFile();

	int getVecSize();
};

#endif
