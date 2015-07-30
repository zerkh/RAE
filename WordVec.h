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
	int								amountOfWords;		//´Ê»ãÁ¿
	int								amountOfStrings;	//´Ê×éÁ¿

public:
	static map<string, Vector*>			m_words;
	static map<string, Vector*>			m_strings;

public:
	WordVec();

	void readFile(Parameter* para, string titleStr);
	void showWords();
	void showStrings();
	bool isInDict(string word);
};

#endif
