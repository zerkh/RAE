#include "Util.h"
#include "Domain.h"
#include "Parameter.h"
#include "WordVec.h"
#include "MutiThreading.h"

using namespace std;

void train( worker_arg_t *arg );
void dev( worker_arg_t *arg );
void test( worker_arg_t *arg );

int main(int argc, char* argv[])
{
	lbfgsfloatval_t start, end;

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

	start = clock();
	RAE* srcRAE = new RAE(para, srcWords, SL);
	end = clock();
	cout << "The time of initial source language RAE is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	start = clock();
	RAE* tgtRAE = new RAE(para, tgtWords, TL);
	end = clock();
	cout << "The time of initial target language RAE is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	bool isDev = atoi(para->getPara("IsDev").c_str());
	bool isTrain = atoi(para->getPara("IsTrain").c_str());
	bool isTest = atoi(para->getPara("IsTest").c_str());

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

	cout << "Start training RAES......" << endl << endl;
	start = clock();
	if(isTrain || isDev)
	{
		srcRAE->training();
		tgtRAE->training();
	}
	end = clock();
	cout << "The time of training RAEs is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	start = clock();
	if(isTest)
	{
		srcRAE->loadWeights(para);
		tgtRAE->loadWeights(para);
	}
	end = clock();
	cout << "The time of RAEs' weights is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	//初始化
	worker_arg_t *wargs = new worker_arg_t[thread_num];

	//分发文件
	for (int wid = 0; wid < thread_num; wid++)
	{
		start = clock();
		cout << "Initialize " << v_domains[wid] << "......" << endl << endl;
		
		wargs[wid].m_id = wid;
		wargs[wid].domainName = v_domains[wid];
		wargs[wid].domain = new Domain(para, v_domains[wid], srcRAE, tgtRAE);

		end = clock();
		cout << "Finish initializing " << v_domains[wid] << " in " << (end-start)/CLOCKS_PER_SEC << "s" << endl << endl;
	}

	if(isTrain)
	{
		Start_Workers(train, wargs, thread_num);
	}
	else if(isDev)
	{
		Start_Workers(dev, wargs, thread_num);
	}
	else if(isTest)
	{
		Start_Workers(test, wargs, thread_num);
	}

	return 0;
}

/************************************************************************/
/* 功能：启动进程开始翻译                                               */
/* 参数：worker_arg_t *arg，输入参数                                    */
/* 返回：vector<string>                                                 */
/************************************************************************/
void train(worker_arg_t* arg)
{
	lbfgsfloatval_t start, end;

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
}

void test(worker_arg_t* arg)
{
	lbfgsfloatval_t start, end;

	Domain* d = arg->domain;
	cout << "Processing " << d->domainName << "......" << endl << endl;

	cout << "Loading " + d->domainName + " testing data..." << endl << endl;
	start = clock();
	d->loadTestingData();
	end = clock();
	cout << "The time of loading " + d->domainName + " testing data is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	cout << "Starting testing " + d->domainName + "..." << endl << endl;
	start = clock();
	d->test();
	end = clock();
	cout << "The time of testing " + d->domainName + " is " << (end-start)/CLOCKS_PER_SEC << endl << endl;
}

void dev( worker_arg_t *arg )
{
	lbfgsfloatval_t start, end;

	Domain* d = arg->domain;
	cout << "Processing " << d->domainName << "......" << endl << endl;

	cout << "Loading " + d->domainName + " dev data..." << endl << endl;
	start = clock();
	d->loadTrainingData();
	end = clock();
	cout << "The time of loading " + d->domainName + " dev data is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

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
