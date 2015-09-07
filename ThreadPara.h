#ifndef THREADPARA
#define THREADPARA
#include "Util.h"
#include "RAE.h"
#include "Domain.h"

class ThreadPara
{
public:
	lbfgsfloatval_t lossVal;
	int instance_num;
	lbfgsfloatval_t* g_RAE;
	lbfgsfloatval_t* g_RM;

public:
	Domain* d;

	ThreadPara()
	{
		d = NULL;
		g_RM = NULL;
		g_RAE = NULL;
	}

	ThreadPara(const ThreadPara& threadpara)
	{
		this->d = threadpara.d;
		this->g_RAE = threadpara.g_RAE;
		this->g_RM = threadpara.g_RM;
		lossVal = threadpara.lossVal;
		instance_num = threadpara.instance_num;
	}

	ThreadPara& operator =(const ThreadPara& threadpara)
	{
		this->d = threadpara.d;
		this->g_RAE = threadpara.g_RAE;
		this->g_RM = threadpara.g_RM;
		this->lossVal = threadpara.lossVal;
		this->instance_num = threadpara.instance_num;

		return *this;
	}
};

class RAEThreadPara:public ThreadPara
{
public:
	RAE* cRAE;
	lbfgsfloatval_t* g;

	RAEThreadPara()
	{
		cRAE = NULL;
	}

	RAEThreadPara(const RAEThreadPara& threadpara)
	{
		this->cRAE = threadpara.cRAE;
		this->g = threadpara.g;
		lossVal = threadpara.lossVal;
		instance_num = threadpara.instance_num;
	}

	RAEThreadPara& operator =(const RAEThreadPara& threadpara)
	{
		this->cRAE = threadpara.cRAE;
		this->g = threadpara.g;
		this->lossVal = threadpara.lossVal;
		this->instance_num = threadpara.instance_num;

		return *this;
	}
};

#endif
