#ifndef myMatrix_h_
#define myMatrix_h_


#include <vector>
#include <ostream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <complex>



//Muzu dodelat nejake fail stavy


//forward declaration for abyility to have referance to obejct
class Matrix;

//fake ref to a point in the matrix, made to eliminate code from library
class Matrix_point_ref {
public:
	//basic contructor, just take referance and coordinates
	Matrix_point_ref(Matrix& m, size_t colm, size_t row) : m(m), colm(colm), row(row){ }
	//forward declaration, uses elements from matrix
	void operator=(float num);
	void operator+=(float num);

private:
	Matrix& m;
	size_t colm;
	size_t row;


};

//fake ref for collum, made to eliminate code from library
class Matrix_col_ref {
public:
	//baic constructor, just take referance and coordinates
	Matrix_col_ref(Matrix& m, size_t colm) : m(m), colm(colm) { }
	//forward declaration, uses elements from matrix
	void operator=(float num);
	void setZero();
	

private:
	Matrix& m;
	size_t colm;
};

//fake ref for row
class Matrix_row_ref {
public:
	//baic constructor, just take referance and coordinates
	Matrix_row_ref(Matrix& m, size_t row) : m(m) , row(row){}
	//forward declaration, uses elements from matrix
	void operator=(float num);
	//for norm calculation
	float colABS();

private:
	Matrix& m;
	size_t row;

};

//The mtarix class
class Matrix {
public:
	//if there are no parametrs, just create blank matix
	Matrix() : colm{ 0 }, row{0} {}
	//if sizes are given, create matrix of that size with zeros
	Matrix(std::size_t colm, std::size_t row) : colm(colm), row(row), matrix_val(colm*row, 0) {}
	//Classic copy constructor, just copy every value
	Matrix(const Matrix&) = default;
	//move constructor, we null the incoming matrix, number of cols and rows, so it's not useable
	Matrix(Matrix&& mat) :colm{ mat.colm }, row{ mat.row }, matrix_val{ std::move(mat.matrix_val) } {mat.colm = mat.row = 0; }
	//classic copy construction by =
	Matrix& operator=(const Matrix&) = default;
	//same as move constrution without =
	Matrix& operator=(Matrix&& mat) 
	{
		colm = mat.colm;
		row = mat.row;
		matrix_val = std::move(mat.matrix_val);
		mat.colm = mat.row = 0;
		return *this;
	}

	//return new matrix that is the transposition of this matrix
	Matrix transpose() const
	{
		//cols and rows are switch in transpositon
		Matrix tmp(row, colm);

		//go through each element of this matrix and based on it change the proper element in new matrix
		for (std::size_t i = 0; i < colm; i++)
		{
			for (std::size_t j = 0; j < row; j++) {tmp.change_at(j, i, at(i, j));}
		}
		return tmp; //return the new matrix
	}

	//Fake refferance to a "object" in matrix, for col and for point
	Matrix_col_ref col(std::size_t i) { return Matrix_col_ref(*this, i); }
	Matrix_row_ref rowReff(std::size_t i) { return Matrix_row_ref(*this, i); }
	Matrix_point_ref coeffRef(std::size_t colm, std::size_t row) { return Matrix_point_ref(*this, colm, row); }


	//uses tanhf on every point in matrix
	void Tanh_M()
	{
			for (std::size_t i = 0; i < colm; i++) 
			{
				for (std::size_t j = 0; j < row; j++) {
					//change each element with tanhf of its self (hyperbolic tangens)
					change_at(i, j, tanhf(at(i, j))); 
				}
			}
		}

	//returns the matrix sizes  
	std::size_t cols() const { return colm; }
	std::size_t rows() const { return row; }

	
	//return the value of a point
	float at(std::size_t colm_c, std::size_t row_c) const { return matrix_val[row_c * colm + colm_c]; } //we don't operate on 2D array but a single vector so the rows and cols are imaginary
	//change the value of a point, by the given number
	void change_at(std::size_t colm_c, std::size_t row_c, float num) {matrix_val[row_c * colm + colm_c] = num;}

	//populates matrix with random values
	void setRandom() 
	{
		for(std::size_t i = 0; i<colm; i++)
		{
			for (std::size_t j = 0; j < row; j++) 
			{
				//for each element create a random number and by dividing it by RAND_MAX, the value will be in range (0.0,1.0)
				float randF = (float)std::rand() / (float)RAND_MAX;
				//to create if the number should have -1, create new number and split the outcome to 2 halfs
				int randInt = std::rand();
				if (randInt > RAND_MAX / 2) { randF = randF * (-1); }
				change_at(i, j, randF);
			}
		}
	}

	//if the coming matrix is not the proper size, just absorb the data the other matrix have
	void AbsorbData(const Matrix& m) {
		//change elements, based on the position
		for (std::size_t i = 0; i < m.cols(); i++)
		{
			for (std::size_t j = 0; j < m.rows(); j++) 
			{
				if (i < colm && j < row) {change_at(i, j, m.at(i, j));}
			}
		}
	}

	//same but for absorbing vectors
	void AbsorbData(const std::vector<float>& m)
	{
		for (std::size_t i = 0; i < m.size(); i++)
		{
			if (i < colm) {change_at(i, 0, m[i]);}
		}
	}

	//This matrix type is not useable for other operations, so this gives option to just get the vector data
	std::vector<float> GetDataBack()
	{
		return matrix_val;
	}
	
private:
	//what makes matrix a matrix is just the sizes, it opearates on a single vector
	std::size_t colm, row;
	std::vector<float> matrix_val;
};

//= operator for the point reff just calls change and the position reff is pointing to 
inline void Matrix_point_ref::operator=(float num) {m.change_at(colm, row, num); }

//+= is a combination of using change and at of the matrix
inline void Matrix_point_ref::operator+=(float num) {m.change_at(colm, row, m.at(colm,row) + num); }

//= operator just goes through the col and uses change for each position
inline void Matrix_col_ref::operator=(float num) 
{
	for (std::size_t i = 0; i < m.rows(); i++) {m.change_at(colm, i, num); }
}

//same as = but with set number that is assigned
inline void Matrix_col_ref::setZero() 
{
	for (std::size_t i = 0; i < m.rows(); i++) {m.change_at(colm, i, 0); }
}


//= operator just goes through the row and uses change for each position
inline void Matrix_row_ref::operator=(float num)
{
	for (std::size_t i = 0; i < m.cols(); i++) { m.change_at(i, row, num); }
}

//calculate the scalar of itself
inline float Matrix_row_ref::colABS()
{
	float ABS = 0;
	for (std::size_t i = 0; i < m.cols(); i++) { ABS += (m.at(i, row) * m.at(i,row)); }
	return ABS;
}

//matrix multiplication
Matrix operator*(const Matrix& m1, const Matrix& m2)
{
	//for propper multiplication
	if (m1.cols() == m2.rows()) 
	{
		//create new matrix with propper size
		Matrix mat(m2.cols(), m1.rows());
		
		//O(n^3) algorithm, just basic multiplication by parts
		for (std::size_t i = 0; i < m1.rows(); i++)
		{
			for (std::size_t j = 0; j < m2.cols(); j++) 
			{
				float tmp = 0;

				for (std::size_t k = 0; k < m2.rows(); k++) {tmp += m1.at(k, i) * m2.at(j, k); }

				mat.coeffRef(j, i) = tmp;
			}
		}
		return mat;
	}
	//if the sizes are not correct, return blank matrix
	else { return Matrix(); }
}

//matrix subbtraction
Matrix operator-(const Matrix& m1, const Matrix& m2) 
{
	//sizes, need to be same
	if (m1.cols() == m2.cols() && m1.rows() == m2.rows())
	{
		//new matrix with same size
		Matrix tmp(m1.cols(), m1.rows());

		//poppulate it with the subbtraction of the 2 matrices
		for (std::size_t i = 0; i < m1.cols(); i++) 
		{
			for (std::size_t j = 0; j < m1.rows(); j++) { tmp.coeffRef(i, j) = (m1.at(i, j) - m2.at(i, j)); }
		}
		return tmp;
	}
	//if the sizes differ, return blank matrix
	else { return Matrix(); }
}

//operator for printing matrix out
std::ostream& operator<<(std::ostream& o, const Matrix& m)
{
	//print each number and close rows with brackets
	for (std::size_t i = 0; i < m.rows(); i++)
	{
		o << "[";

		for (std::size_t j = 0; j < m.cols(); j++) { o << m.at(j, i) << ','; }

		o << "]";
		o << '\n';
	}
	return o;
}

#endif