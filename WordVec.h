#ifndef WORDVEC_H
#define WORDVEC_H

#include <iostream>
#include <map>
#include <cstdlib>
#include "Parameter.h"
#include "Vec.h"

using namespace std;

class WordVec
{
private:
	typedef pair<string, Vector*>	Word;
	int								amountOfWords;		//�ʻ���
	int								amountOfStrings;	//������

public:
	static map<string, Vector*>			m_words;
	static map<string, Vector*>			m_strings;

public:
	WordVec();

	void readFile(Parameter* para);
	void showWords();
	void showStrings();
};

#endif
