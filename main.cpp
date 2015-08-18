#include <iostream>
#include <ctime>
#include <sstream>
#include "Domain.h"
#include "Parameter.h"
#include "WordVec.h"
#include "MutiThreading.h"
#include <vector>

using namespace std;

void work( worker_arg_t *arg );

int main(int argc, char* argv[])
{
	double start, end;

	start = clock();
	Parameter* para = new Parameter(argv[1]);
	end = clock();
	cout << "The time of reading parameter is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	start = clock();
	WordVec* srcWords = new WordVec();
	srcWords->readFile(para, "MixedDomainSrc");
	end = clock();
	cout << "The time of reading srcWordVec is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	start = clock();
	WordVec* tgtWords = new WordVec();
	tgtWords->readFile(para, "MixedDomainTgt");
	end = clock();
	cout << "The time of reading tgtWordVec is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	// thread_num = 1
	int thread_num = atoi( para->getPara("THREAD_NUM").c_str() );
	vector<string> v_domains;

	string domainLine = para->getPara("DomainList");
	string domainName;
	stringstream ss(domainLine);
	int count = 0;
	while(ss >> domainName)
	{
		if(count == thread_num)
		{
			break;
		}
		count++;
		v_domains.push_back(domainName);
	}

	//初始化
	worker_arg_t *wargs = new worker_arg_t[thread_num];

	//分发文件
	for (int wid = 0; wid < thread_num; wid++)
	{
		start = clock();
		cout << "Initialize " << v_domains[wid] << "......" << endl << endl;
		
		wargs[wid].m_id = wid;
		wargs[wid].domainName = v_domains[wid];
		wargs[wid].domain = new Domain(para, v_domains[wid], srcWords, tgtWords);

		end = clock();
		cout << "Finish initializing " << v_domains[wid] << " in " << (end-start)/CLOCKS_PER_SEC << "s" << endl << endl;
	}

	Start_Workers(work, wargs, thread_num);

	return 0;
}

/************************************************************************/
/* 功能：启动进程开始翻译                                               */
/* 参数：worker_arg_t *arg，输入参数                                    */
/* 返回：vector<string>                                                 */
/************************************************************************/
void work(worker_arg_t* arg)
{
	double start, end;

	Domain* d = arg->domain;
	cout << "Processing " << d->domainName << "......" << endl << endl;
	
	cout << "Loading " + d->domainName + " training data..." << endl << endl;
	start = clock();
	d->loadTrainingData();
	end = clock();
	cout << "The time of loading " + d->domainName + " training data is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	cout << "Starting training " + d->domainName + " ..." << endl << endl;
	start = clock();
	d->training();
	end = clock();
	cout << "The time of training " + d->domainName + " is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	cout << "Starting testing " + d->domainName + "..." << endl << endl;
	start = clock();
	d->test();
	end = clock();
	cout << "The time of testing " + d->domainName + " is " << (end-start)/CLOCKS_PER_SEC << endl << endl;
}
