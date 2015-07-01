#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
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

#endif