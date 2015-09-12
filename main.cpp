#include "Util.h"
#include "Domain.h"
#include "Parameter.h"
#include "WordVec.h"
#include "MutiThreading.h"
#include "MixedDomain.h"

using namespace std;

void raeTrain(worker_arg_t* arg);

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

/*
	start = clock();
	WordVec* tgtWords = new WordVec();
	tgtWords->readFile(para, "MixedDomainTgt");
	end = clock();
	cout << "The time of reading tgtWordVec is " << (end-start)/CLOCKS_PER_SEC << endl << endl;*/

	start = clock();
	RAE* srcRAE = new RAE(para, srcWords, SL);
	end = clock();
	cout << "The time of initial source language RAE is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

/*
	start = clock();
	RAE* tgtRAE = new RAE(para, tgtWords, TL);
	end = clock();
	cout << "The time of initial target language RAE is " << (end-start)/CLOCKS_PER_SEC << endl << endl;*/

	bool isDev = atoi(para->getPara("IsDev").c_str());
	bool isTrain = atoi(para->getPara("IsTrain").c_str());
	bool isTest = atoi(para->getPara("IsTest").c_str());

/*
	worker_arg_t* warg = new worker_arg_t[2];
	warg[0].rae = srcRAE;
	warg[0].m_id = 0;
	warg[1].rae = tgtRAE;
	warg[1].m_id = 1;*/

	cout << "Start training RAES......" << endl << endl;
	start = clock();
	if(isTrain || isDev)
	{
		//Start_Workers(raeTrain, warg, 2);
		srcRAE->training();
	}
	end = clock();
	cout << "The time of training RAEs is " << (end-start)/CLOCKS_PER_SEC << endl << endl;

	if(isTest)
	{
		start = clock();
		srcRAE->loadWeights(para);
/*		tgtRAE->loadWeights(para);*/
		end = clock();
		cout << "The time of RAEs' weights is " << (end-start)/CLOCKS_PER_SEC << endl << endl;
	}

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

	vector<Domain*> domains;
	//分发文件
	for (int wid = 0; wid < thread_num; wid++)
	{
/*		Domain* domain = new Domain(para, v_domains[wid], srcRAE, tgtRAE);*/
		Domain* domain = new Domain(para, v_domains[wid], srcRAE);
		domains.push_back(domain);
	}


/*	MixedDomain mixedDomains(para, domains, srcRAE, tgtRAE);*/
	MixedDomain mixedDomains(para, domains, srcRAE);
	if(isTrain)
	{
		mixedDomains.training();
	}
	else if(isDev)
	{
		mixedDomains.training();
		mixedDomains.testing();
	}
	else if(isTest)
	{
		mixedDomains.testing();
	}

	return 0;
}

void raeTrain(worker_arg_t* arg)
{
	arg->rae->training();
}
