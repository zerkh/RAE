#ifndef VEC_H
#define VEC_H

#include <iostream>
#include "Util.h"

using namespace std;

class Vector
{
private:
	double**	vec;
	int			row;
	int			col;

public:
	Vector(int row, int col);

	Vector();

	Vector* multiply(Vector* sec_Vec, bool is_Transpose);

	double getValue(int row, int col);

	void setValue(int row, int col, double value);

	int getRow();

	int getCol();

	void transpos();

	void showVector();

	void randInitVector();

	Vector* concat(Vector* secVec);

	Vector* add(Vector* secVec);

	~Vector();
};

#endif
