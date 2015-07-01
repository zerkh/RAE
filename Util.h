#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <string>
#include <sstream>
using namespace std;

string strip_str(string str)
{
	string newStr = "";

	for(int i = 0; i < str.size(); i++)
	{
		if(str[i] == '\t')
		{
			continue;
		}

		newStr += str[i];
	}

	return newStr;
}

#endif