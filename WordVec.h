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
	typedef pair<string, MatrixXd>	Word;
	int								amountOfWords;		//´Ê»ãÁ¿
	int								amountOfStrings;	//´Ê×éÁ¿

public:
	static map<string, MatrixXd>			m_words;
	static map<string, MatrixXd>			m_strings;

public:
	WordVec();

	void readFile(Parameter* para, string titleStr);
	void showWords();
	void showStrings();
	bool isInDict(string word);
};

#endif
