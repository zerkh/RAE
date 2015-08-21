#ifndef CONSTDEF
#define CONSTDEF
#include <map>
#include <utility>
#include <cmath>

using namespace std;

const int BASED_NODE = 0;
const int COMBINED_NODE = 1;
const int REC_NODE = 2;

typedef pair<int, int> span;

//Erec
const double ALPHA = 1 * pow(10, -20);
//Ereo
const double BETA = 1 * pow(10, -20);
//Edis
const double GAMMA = 1 * pow(10, -20);
//Decay
const double ZETA = 1 * pow(10, -20);
//learning rate
const double RATE = 1 * pow(10, -20);

#endif
