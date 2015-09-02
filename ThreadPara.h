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
	lbfgsfloatval_t* g;
};

class RAEThreadPara:public ThreadPara
{
public:
	RAE* cRAE;

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

class RMThreadPara:public ThreadPara
{
public:
	Domain* d;

	RMThreadPara()
	{
		d = NULL;
	}

	RMThreadPara(const RMThreadPara& threadpara)
	{
		this->d = threadpara.d;
		this->g = threadpara.g;
		lossVal = threadpara.lossVal;
		instance_num = threadpara.instance_num;
	}

	RMThreadPara& operator =(const RMThreadPara& threadpara)
	{
		this->d = threadpara.d;
		this->g = threadpara.g;
		this->lossVal = threadpara.lossVal;
		this->instance_num = threadpara.instance_num;

		return *this;
	}
};

#endif
