#ifndef WORDVEC_H
#define WORDVEC_H

#include "Util.h"
#include "Parameter.h"
#include "Eigen/Core"

using namespace std;
using namespace Eigen;

class WordVec
{
private:
	typedef pair<string, MatrixLBFGS>	Word;
	int								amountOfWords;		//´Ê»ãÁ¿
	int								amountOfStrings;	//´Ê×éÁ¿

public:
	static map<string, MatrixLBFGS>			m_words;
	static map<string, MatrixLBFGS>			m_strings;

public:
	WordVec();

	void readFile(Parameter* para, string titleStr);
	void showWords();
	void showStrings();
	bool isInDict(string word);
};

#endif
