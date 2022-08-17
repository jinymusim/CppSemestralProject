#ifndef my_Nural_
#define my_Nural_

#include <vector>
#include <memory>
#include <string>
#include <math.h>
#include <functional>
#include <iostream>
#include <fstream>
#include "myMatrix.hpp"

//Neuralni sit vzata z:
//https://www.geeksforgeeks.org/ml-neural-network-implementation-in-c-from-scratch/
//upravena pro eliminaci knihovny Eigen a pridany doplnovaci funkce.
//upravena pro eliminaci chyb

//derivative of tan funkticon
inline float tanDerv(float x)
{
	return (1 - tanhf(x) * tanhf(x));
}

//Neural network class, holds neurnos and weight matrices
class NeuralNetwork {
public:

	//if no parametrs are added just initilize all vectors by clearing them.
	explicit NeuralNetwork(float learning_volitility = 0.005) : learning_volitility(learning_volitility)
	{

		neurons.clear();
		cached.clear();
		deltas.clear();
		matrix_vector.clear();
		matrix_description.clear();

	}

	//if filename is added as construtor parametr, load network weights from file, set initial neuron level
	NeuralNetwork(const std::string& filename, float learning_volitility = 0.005) : learning_volitility(learning_volitility)
	{
		//initilize all vectors by clearing them
		neurons.clear();
		cached.clear();
		deltas.clear();
		matrix_vector.clear();
		matrix_description.clear();

		//load the weights
		loadNetwork(filename);

		//create the neuron levels
		for (std::size_t i = 0; i < matrix_description.size(); i++) 
		{
			//if its last layer, leave just the output
			if (i == matrix_description.size() - 1) { neurons.push_back(std::make_unique<Matrix>(matrix_description[i],1)); }
			//if the layer is not last add one neuron to keep constant 1, so the weight doesn't go to up.
			else { neurons.push_back(std::make_unique<Matrix>(matrix_description[i] + 1,1)); }

			//for each neuron create cache and deltas, to keep data for backtracking and error calculating
			cached.push_back(std::make_unique<Matrix>(neurons[i]->cols(),1));
			deltas.push_back(std::make_unique<Matrix>(neurons[i]->cols(),1));

			//set the 1 for neurons, exclude last layer
			if (i != matrix_description.size()-1)
			{
				neurons.back()->coeffRef(matrix_description[i],0) = 1.0;
				cached.back()->coeffRef(matrix_description[i],0) = 1.0;
			}
		}
	}

	//if there is no file to fill the weights, but user provides network description, set the weights as random
	NeuralNetwork(std::vector<std::size_t>& network_depth, float learning_volitility = 0.005) : learning_volitility(learning_volitility), matrix_description(network_depth)
	{
		//go through all the description
		for (std::size_t i = 0; i < matrix_description.size(); i++) 
		{
			//cretae the neurons, if the layer isn't last and one neuron to hold 1.
			if (i == matrix_description.size() - 1) { neurons.push_back(std::make_unique<Matrix>(matrix_description[i],1)); }
			else {neurons.push_back(std::make_unique<Matrix>(matrix_description[i] + 1,1));} 

			//based on neuron create the cache and delatas, to keep data for backtracking and error calculation
			cached.push_back(std::make_unique<Matrix>(neurons[i]->cols(),1)); 
			deltas.push_back(std::make_unique<Matrix>(neurons[i]->cols(),1));

			//set the 1 for neurons, exclude last layer
			if (i != matrix_description.size()-1) 
			{
				neurons.back()->coeffRef(matrix_description[i],0) = 1.0;
				cached.back()->coeffRef(matrix_description[i],0) = 1.0;
			}
			//there are 1 less matrices than neuron layres, they represent the weights between neurons
			if (i > 0) 
			{
				//if its not the last matrix (with layer smaller by 1), create them 1 bigger to take the extra neuron
				if (i != matrix_description.size() - 1) 
				{
					matrix_vector.push_back(std::make_unique<Matrix>(matrix_description[i] + 1, matrix_description[i-1] + 1));
					//set them to random value, we don't have info about the data
					matrix_vector.back()->setRandom();
					//set the last col, to compensate for the extra neuron
					matrix_vector.back()->col(matrix_description[i]).setZero();
					matrix_vector.back()->coeffRef(matrix_description[i], matrix_description[i-1]) = 1.0;
				}
				//if its last matrix it needs to fit the outup, but there is no extra neuron to trip us
				else 
				{
					matrix_vector.push_back(std::make_unique<Matrix>(matrix_description[i], matrix_description[i-1]+ 1));
					matrix_vector.back()->setRandom();
				}
			}
		}
	}

	//forward computation
	void propagateForward(Matrix& input) 
	{
		//absorb the input data, we can't use the input it self, because of the extra neuron that holds 1
		cached.front()->AbsorbData(input);
		neurons.front()->AbsorbData(input);

		//calculate the matrix multiplication and go to next level of neurons
		for (std::size_t i = 1; i < matrix_description.size(); i++) 
		{
			(*cached[i]) = (*cached[i - 1]) * (*matrix_vector[i - 1]);

		}
		//callculate the hyperbolic tangent to return to -1-+1, don't recalculate input
		for (std::size_t i = 1; i < matrix_description.size(); i++) 
		{
			Matrix tmp = *cached[i];
			tmp.Tanh_M();
			neurons[i] = std::make_unique<Matrix>(std::move(tmp));
			//for the hidden layers add back the 1
			if (i != matrix_description.size() - 1) {neurons[i]->coeffRef(matrix_description[i], 0) = 1;}
		}

	}

	//calculate the error from the output
	void calcErrors(Matrix& output) 
	{
		//calculate the back error by simple subtraction
		(*deltas.back()) = output - (*neurons.back());

		//the other layers are calculated by backwords calculation
		for (std::size_t i = matrix_description.size() - 2; i > 0; i--) 
		{
			(*deltas[i]) = (*deltas[i + 1]) * (matrix_vector[i]->transpose());
		}

	}

	//update the matrices weights, based on the error and learning rate
	void update() 
	{
		//once again, the number of matrices is 1 less than the number of layers
		for (std::size_t i = 0; i < matrix_description.size() - 1; i++) 
		{
			//if the matrix is not the last, we need to exclude part to not change the extra 1 neuron
			if (i != matrix_description.size() - 2) 
			{

				for (std::size_t j = 0; j < matrix_vector[i]->cols()-1; j++) 
				{
					for (std::size_t k = 0; k < matrix_vector[i]->rows(); k++) 
					{
						//add the learning to the weight
						float tmp = learning_volitility * (deltas[i + 1]->at(j, 0)) * (tanDerv(cached[i + 1]->at(j, 0))) * (neurons[i]->at(k, 0)); //zde chyba
						matrix_vector[i]->coeffRef(j, k) += tmp;
					}
				}
			}
			//if it's last we don't need to worry about the extra 1 neuron
			else 
			{
				for (std::size_t j = 0; j < matrix_vector[i]->cols(); j++) 
				{
					for (std::size_t k = 0; k < matrix_vector[i]->rows(); k++) 
					{
						//add the learning
						matrix_vector[i]->coeffRef(j, k) += learning_volitility * deltas[i + 1]->at(j,0) * tanDerv(cached[i + 1]->at(j,0)) * neurons[i]->at(k,0);
					}
				}
			}
		}
	}

	//the whole back propagation is just error calculation and weight update
	void propagateBackward(Matrix& output) 
	{
		calcErrors(output);
		update();
	}

	//train the network on the given data
	void train(std::vector<Matrix*>& input, std::vector<Matrix*>& output) 
	{
		//go through the data and propagate Forward and Backward
		for (std::size_t i = 0; i < input.size(); i++) 
		{
			std::cout << "Input to neural network is : " << *input[i] << std::endl;
			propagateForward(*input[i]);
			std::cout << "Expected output is : " << *output[i] << std::endl;
			std::cout << "Output produced is : " << *neurons.back() << std::endl;
			propagateBackward(*output[i]);
			std::cout << "MSE : " << std::sqrt(deltas.back()->rowReff(0).colABS() / deltas.back()->cols()) << std::endl;
		}
	}
	//Test the data, but no training based on the data
	void test(std::vector<Matrix*>& input, std::vector<Matrix*>& output) 
	{
		for (std::size_t i = 0; i < input.size(); i++) 
		{
			std::cout << "Input to neural network is : " << *input[i] << std::endl;
			propagateForward(*input[i]);
			std::cout << "Expected output is : " << *output[i] << std::endl;
			std::cout << "Output produced is : " << *neurons.back() << std::endl;
		}
	}

	//return result for given input
	Matrix reuslts(Matrix& input) 
	{
		//propagate and return last layer, which is output
		propagateForward(input);
		return *neurons.back();
	}


	//same but for input as vector
	std::vector<float> results(std::vector<float>& input)
	{
		//let the input be absorbed	by the matrix
		Matrix m(input.size(), 1);
		m.AbsorbData(input);
		//propagate and return data as vector
		propagateForward(m);
		return neurons.back()->GetDataBack();

	}

	//Save the network to a given filename
	void saveLearnedNetwork(const std::string& filename) 
	{
		//open the file truncate it
		std::fstream fout(filename, std::fstream::in | std::fstream::out | std::fstream::trunc);
		//string for description
		std::string network_des;

		for (std::size_t i = 0; i < matrix_description.size(); i++) 
		{
			network_des += (std::to_string(matrix_description[i]) + ',');
		}
		//write the describtion
		fout << network_des << std::endl;

		//go through the matrix and write down the rows
		for (std::size_t i = 0; i < matrix_description.size() - 1; i++) 
		{
			for (std::size_t j = 0; j < matrix_vector[i]->rows(); j++) 
			{
				std::string row_str;
				for (std::size_t k = 0; k < matrix_vector[i]->cols(); k++)
				{
					row_str += (std::to_string(matrix_vector[i]->at(k,j)) + ',');
				}
				fout << row_str << std::endl;
			}
		}

	}

	//load the network from a given file
	void loadNetwork(const std::string& filename) 
	{
		//open the file
		std::fstream fin(filename);
		std::string network_des;

		//first line is about network shape
		std::getline(fin, network_des);

		//parse the numbers 
		int temp = 0;
		for (std::size_t i = 0; i < network_des.size(); i++) 
		{
			//if separation symbol is found push back the number
			if (network_des[i] == ',' || network_des[i] == ';')
			{
				matrix_description.push_back(temp);
				temp = 0;
			}
			//if not recalculate the number
			else
			{
				temp = temp * 10 + (network_des[i] - '0');
			}
		}
		//if there is no separation at the end, but there still is processed number save it
		if (temp != 0) { matrix_description.push_back(temp); }

		//now that the description is known load the individual values
		for (std::size_t i = 1; i < matrix_description.size(); i++)
		{
			//one larger to compensate for the extra neuron with 1
			if (i != matrix_description.size() - 1) { matrix_vector.push_back(std::make_unique<Matrix>(matrix_description[i] + 1, matrix_description[i-1] + 1)); }
			//if the matrix is last, we don't add the 1 for the extra neuron
			else { matrix_vector.push_back(std::make_unique<Matrix>(matrix_description[i], matrix_description[i-1] +1)); }

			for (std::size_t j = 0; j < matrix_vector.back()->rows(); j++)
			{
				for (std::size_t k = 0; k < matrix_vector.back()->cols(); k++)
				{
					//load individual cell and get the float number
					std::string num;
					std::getline(fin, num, ',');

					std::size_t p2 = 0;
					float numF = std::stof(num.c_str(), &p2);
					//update the matrix with the float
					if (p2 != 0) { matrix_vector.back()->coeffRef(k,j) = numF; }
					else { matrix_vector.back()->coeffRef(k, j) = 0; }
				}
			}
		}
	}


private:
	//network atributes
	std::vector<std::unique_ptr<Matrix>> neurons; //neurons for calculating
	std::vector<std::unique_ptr<Matrix>> cached; //to hold the base state
	std::vector<std::unique_ptr<Matrix>> deltas; //to hold the individual errors
	std::vector<std::unique_ptr<Matrix>> matrix_vector; //to hold the matrices of weights between neuron layers
	float learning_volitility; //how volitile is the learning
	std::vector<size_t> matrix_description; //how the network looks

};

#endif 

