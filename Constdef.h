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
const double ALPHA = 2 * pow(10, -8);
//Ereo
const double BETA = 6 * pow(10, -8);
//Edis
const double GAMMA = 6* pow(10, -8);
//Decay
const double ZETA = 1 * pow(10, -6);
//learning rate
const double RATE = 5 * pow(10, -8);

#endif
