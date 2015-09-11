#include "MutiThreading.h"

#include <iostream>

using namespace std;

void Wait(int worker_num)
{
	for (int i = 0; i < worker_num; i++)
	{
		int status;
		wait(&status);
		if (!WIFEXITED(status))
		{
			cerr << "Warning: abnormal terminated!" << endl;
		}
	}
}




void Start_Workers(worker_t worker, worker_arg_t *args, int worker_num)
{
	for (int i = 1; i < worker_num; i++)
	{
		if (!fork())
		{ 
			worker(&args[i]);
			//cout << "Worker " << args[i].wid << " finishes\n";
			exit(0);
		}
	}
	//Do the first chunck myself
	worker(&args[0]);
	Wait(worker_num - 1);
}
