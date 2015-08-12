#include "Vec.h"
#include <cstdlib>

void Vector::setToZeros()
{
	for(int row = 0; row < getRow(); row++)
	{
		for(int col = 0; col < getCol(); col++)
		{
			this->setValue(row, col, 0);
		}
	}
}

Vector* Vector::copy()
{
	Vector* newVec = new Vector(row, col);

	for(int r = 0; r < row; r++)
	{
		for(int c = 0; c < col; c++)
		{
			newVec->setValue(r, c, this->getValue(r, c));
		}
	}

	return newVec;
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

	for(int i = 0; i < row; i++)
	{
		for(int j = 0; j < col; j++)
		{
			vec[i][j] = 0;
		}
	}
}

Vector::~Vector()
{
	for(int i = 0; i < row; i++)
	{
		delete []vec[i];
	}
	
	delete[] vec;
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

Vector* Vector::multiply(Vector* sec_Vec, bool is_Transpose)
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
			exit(-1);
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
			exit(-1);
		}
	}
}

//ת��
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

void Vector::showVector()
{
	for(int row = 0; row < this->getRow(); row++)
	{
		for(int col = 0; col < this->getCol(); col++)
		{
			cout << getValue(row, col) << " ";
		}

		cout << endl;
	}
}

//�����ʼ�������ֵ
void Vector::randInitVector()
{
	for(int row = 0; row < getRow(); row++)
	{
		for(int col = 0; col < getCol(); col++)
		{
			this->setValue(row, col, getRand());
		}
	}
}

//����ƴ��
Vector* Vector::concat(Vector* secVec)
{
	int newCol = this->getCol() + secVec->getCol();

	Vector* newVec = new Vector(1, newCol);

	for(int col = 0; col < newCol; col++)
	{
		if(col >= this->getCol())
		{
			newVec->setValue(0, col, secVec->getValue(0, col-this->getCol()));
		}
		else
		{
			newVec->setValue(0, col, this->getValue(0, col));
		}
	}

	return newVec;
}

//�����ӷ�
Vector* Vector::add(Vector* secVec)
{
	if(this->getRow() != secVec->getRow() || this->getCol() != secVec->getCol())
	{
		cerr << "The dimension of two vector is not equall!" << endl << endl;
		exit(-1);
	}

	Vector* newVec = new Vector(getRow(), getCol());

	for(int row = 0; row < getRow(); row++)
	{
		for(int col = 0; col < getCol(); col++)
		{
			newVec->setValue(row, col, this->getValue(row, col) + secVec->getValue(row, col));
		}
	}

	return newVec;
}