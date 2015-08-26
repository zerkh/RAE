#ifndef PARAMETER_H
#define PARAMETER_H

#include "Util.h"
using namespace std;

class Parameter
{
private:
	map<string, string>		m_para;

public:
	Parameter(const string& filename)
	{
		cout << "Reading parameter from \"" + filename + "\"" << endl << endl;

		ifstream is(filename.c_str());

		if(!is)
		{
			cerr << "Cannot read the parameter \"" + filename + "\"" << endl << endl;
			exit(-1);
		}

		string line, titleStr, valueStr;
		int pos;

		while(getline(is, line))
		{
			if (line == "" || (pos = line.find("=")) == string::npos)
			{
				continue;
			}

			titleStr = strip_str(line.substr(0, pos));
			pos++;
			valueStr = strip_str(line.substr(pos));

			m_para[titleStr] = valueStr;
		}

		is.close();
	};

	//获得参数
	string getPara(const string& para)
	{
		map<string, string>::iterator m_it = m_para.find(para);

		if(m_it == m_para.end())
		{
			cerr << "Cannot find \"" + para + "\"" << endl << endl;
			exit(-1);
		}

		return m_it->second;
	}

	//显示所有参数
	void showPara()
	{
		map<string, string>::iterator m_it = m_para.begin();

		while(m_it != m_para.end())
		{
			cout << m_it->first << "\t" << m_it->second << endl << endl;
			m_it++;
		}
	}
};

#endif