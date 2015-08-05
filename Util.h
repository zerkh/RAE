#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <Eigen/Core>

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

inline MatrixXd concatMatrix(MatrixXd m1, MatrixXd m2)
{
	MatrixXd m = MatrixXd(m1.rows(), m1.cols()+m2.cols());

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

inline double getRand()
{
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if ( phase == 0 ) 
	{
		do 
		{
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

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

inline vector<string> splitBySpace(string line)
{
	vector<string> subStr;

	string str;
	stringstream ss(line);

	while(ss >> str)
	{
		subStr.push_back(str);
	}

	return subStr;
}

#endif
