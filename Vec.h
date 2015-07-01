#ifndef VEC_H
#define VEC_H

#include <iostream>

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

	void initVector(int row, int col);

	Vector* Multiply(Vector* sec_Vec, bool is_Transpose);

	double getValue(int row, int col);

	void setValue(int row, int col, double value);

	int getRow();

	int getCol();

	void transpos();

	~Vector();
};

#endif
