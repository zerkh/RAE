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

const int EREO = 0;
const int EDIS = 1;
const int CONS = 2;

const int SL = 0;
const int TL = 1;

typedef pair<int, int> span;

typedef Matrix<lbfgsfloatval_t, Eigen::Dynamic, Eigen::Dynamic> MatrixLBFGS;
typedef Matrix<lbfgsfloatval_t, 1, Eigen::Dynamic> VectorLBFGS;

//Erec
static lbfgsfloatval_t ALPHA = 1 * pow(10, -2);

//Ereo
static lbfgsfloatval_t BETA = 8 * pow(10, -2);

//Edis
static lbfgsfloatval_t GAMMA = 6 * pow(10, -2);

//Decay
static lbfgsfloatval_t ZETA = 1 * pow(10, -3);

//Cons
static lbfgsfloatval_t DELTA = 5 * pow(10, -3);

static bool isUpdateRAE = true;
static bool isUpdateRM = true;

#endif
