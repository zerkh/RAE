#include "Vec.h"

Vector::Vector()
{
}

Vector::Vector(int row, int col)
{
	this->row = row;
	this->col = col;

	vec = new double*[row];

	for(int i = 0; i < row; i++)
	{
		vec[i] = new double[col];
	}
}

Vector::~Vector()
{
	for(int i = 0; i < row; i++)
	{
		delete vec[i];
	}
	delete[] vec;

	//cout << "Delete";
}

void Vector::initVector(int row, int col)
{
	this->row = row;
	this->col = col;

	vec = new double*[row];

	for(int i = 0; i < row; i++)
	{
		vec[i] = new double[col];
	}
}

int Vector::getCol()
{
	return col; 
}

int Vector::getRow()
{
	return row;
}

double Vector::getValue(int row, int col)
{
	return vec[row][col];
}

void Vector::setValue(int row, int col, double value)
{
	vec[row][col] = value;
}

Vector* Vector::Multiply(Vector* sec_Vec, bool is_Transpose)
{
	if(is_Transpose)
	{
		if(this->col == sec_Vec->col)
		{
			Vector* gen_Vec = new Vector(this->row, sec_Vec->row);

			for(int i = 0; i < this->row; i++)
			{
				for(int j = 0; j < sec_Vec->row; j++)
				{
					double temp = 0;

					for(int k = 0; k < this->col; k++)
					{
						temp += this->getValue(i, k) * sec_Vec->getValue(j, k);
					}

					gen_Vec->setValue(i, j, temp);
				}
			}

			return gen_Vec;
		}
		else
		{
			cerr << "Dimension Error" << endl;
			return this;
		}
	}
	else
	{
		if(this->col == sec_Vec->row)
		{
			Vector* gen_Vec = new Vector(this->row, sec_Vec->col);

			for(int i = 0; i < this->row; i++)
			{
				for(int j = 0; j < sec_Vec->col; j++)
				{
					double temp = 0;

					for(int k = 0; k < this->col; k++)
					{
						temp += this->getValue(i, k) * sec_Vec->getValue(k, j);
					}

					gen_Vec->setValue(i, j, temp);
				}
			}

			return gen_Vec;
		}
		else
		{
			cerr << "Dimension Error" << endl;
			return this;
		}
	}
}

//×ªÖÃ
void Vector::transpos()
{
	double** trans_vec = new double*[this->col];
	for(int i = 0; i < col; i++)
	{
		trans_vec[i] = new double[row];
	}

	for(int i = 0; i < col; i++)
	{
		for(int j = 0; j < row; j++)
		{
			trans_vec[i][j] = this->vec[j][i];
		}
	}

	for(int i = 0; i < row; i++)
	{
		delete vec[i];
	}
	delete vec;

	this->vec = trans_vec;
	int temp = row;
	row = col;
	col = temp;
}
