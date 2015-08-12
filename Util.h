#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include <vector>

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

//将每行数据根据c1,e1等划分
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

void operator delete(void* p)
{
	if (p)
	{
		free(p);
		p = NULL;
	}
}

#endif
