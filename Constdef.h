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
double ALPHA = 5 * pow(10, -2);
//Ereo
double BETA = 8 * pow(10, -2);
//Edis
double GAMMA = 6 * pow(10, -2);
//Decay
double ZETA = 1 * pow(10, -3);
//learning rate
double RATE = 5 * pow(10, -3);

#endif
