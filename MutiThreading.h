#ifndef MUTITHREADING
#define MUTITHREADING

#include <vector>
#include "Domain.h"
#include "Util.h"
#include <unistd.h>
#include <sys/wait.h>

using namespace std;

class worker_arg_t
{
public:
	int m_id;
	Domain* domain;
	string domainName;
};

typedef void(*worker_t)(worker_arg_t*);
extern void Start_Workers(worker_t worker, worker_arg_t *args, int workerNum);

#endif