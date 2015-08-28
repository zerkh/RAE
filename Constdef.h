#ifndef CONSTDEF
#define CONSTDEF
#include <map>
#include <utility>
#include <cmath>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

const int BASED_NODE = 0;
const int COMBINED_NODE = 1;
const int REC_NODE = 2;

typedef pair<int, int> span;

typedef Matrix<lbfgsfloatval_t, Eigen::Dynamic, Eigen::Dynamic> MatrixLBFGS;
typedef Matrix<lbfgsfloatval_t, 1, Eigen::Dynamic> VectorLBFGS;

//Erec
static lbfgsfloatval_t ALPHA = 5 * pow(10, -2);
//Ereo
static lbfgsfloatval_t BETA = 8 * pow(10, -2);
//Edis
static lbfgsfloatval_t GAMMA = 6 * pow(10, -2);
//Decay
static lbfgsfloatval_t ZETA = 1 * pow(10, -3);
//learning rate
static lbfgsfloatval_t RATE = 5 * pow(10, -3);

#endif
