#ifndef CONSTDEF
#define CONSTDEF
#include <map>
#include <utility>

using namespace std;

const int BASED_NODE = 0;
const int COMBINED_NODE = 1;
const int REC_NODE = 2;

typedef pair<int, int> span;

//重构参数
const double ALPHA = 0.2;
//Ereo
const double BETA = 1.2;
//平方误差
const double GAMMA = 1.2;
//偏置项
const double ZETA = 0;

const double DELTA = 0.2;
//学习速率
const double RATE = 0.65;

#endif
