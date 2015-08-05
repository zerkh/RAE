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
const double ALPHA = 0.2;
//Ereo
const double BETA = 0.6;
//Edis
const double GAMMA = 0;
//Decay
const double ZETA = 0.01;
//learning rate
const double RATE = 0.65;

#endif
