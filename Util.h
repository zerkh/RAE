#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <map>
#include <vector>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "lbfgs.h"
#include "Constdef.h"

using namespace Eigen;
using namespace std;

inline string strip_str(string str)
{
	string newStr = "";

	for(int i = 0; i < str.size(); i++)
	{
		if(str[i] == '\t' || str[i] == '\"')
		{
			continue;
		}

		newStr += str[i];
	}

	return newStr;
}

inline MatrixLBFGS tanh(MatrixLBFGS m)
{
	for(int row = 0; row < m.rows(); row++)
	{
		for(int col = 0; col < m.cols(); col++)
		{
			m(row, col) = (exp(m(row, col))-exp(-m(row, col)))/(exp(m(row, col))+exp(-m(row, col)));
		}
	}

	return m;
}

inline MatrixLBFGS concatMatrix(MatrixLBFGS m1, MatrixLBFGS m2)
{
	MatrixLBFGS m = MatrixLBFGS(m1.rows(), m1.cols()+m2.cols());

	for(int col = 0; col < m.cols(); col++)
	{
		if(col < m1.cols())
		{
			m(0, col) = m1(0, col);
		}
		else if(col >= m1.cols())
		{
			m(0, col) = m2(0, col-m1.cols());
		}
	}

	return m;
}

inline lbfgsfloatval_t getRand()
{
	static lbfgsfloatval_t V1, V2, S;
	static int phase = 0;
	lbfgsfloatval_t X;

	if ( phase == 0 ) 
	{
		do 
		{
			lbfgsfloatval_t U1 = (lbfgsfloatval_t)rand() / RAND_MAX;
			lbfgsfloatval_t U2 = (lbfgsfloatval_t)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} 
	else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;

	return X;
}

inline vector<string> splitBySign(string line)
{
	vector<string> subStr;

	int spos = line.find("ct1=");
	subStr.push_back(line.substr(0, spos-1));

	int epos = line.find("et1=");
	subStr.push_back(line.substr(spos, epos-1-spos));

	spos = epos;
	epos = line.find("ct2=");
	subStr.push_back(line.substr(spos, epos-1-spos));

	spos = epos;
	epos = line.find("et2=");
	subStr.push_back(line.substr(spos, epos-1-spos));

	subStr.push_back(line.substr(epos, line.size()-epos));

	return subStr;
}

#endif
