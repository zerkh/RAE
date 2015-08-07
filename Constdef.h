#ifndef CONSTDEF
#define CONSTDEF
#include <map>
#include <utility>

using namespace std;

const int BASED_NODE = 0;
const int COMBINED_NODE = 1;
const int REC_NODE = 2;

typedef pair<int, int> span;

//Erec
static double ALPHA = 0.2;
//Ereo
static double BETA = 0.01;
//Edis
static double GAMMA = 0;
//Decay
static double ZETA = 0.001;
//learning rate
static double RATE = 0.2;

#endif
