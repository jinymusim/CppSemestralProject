#ifndef my_CSV_
#define my_CSV_

#include <vector>
#include <string>
#include <ostream>
#include <fstream>
#include <iostream>
#include "NeuralNetwork.hpp"


//simple loading of data
inline void load_data(const std::string& filname, std::vector<Matrix*>& data) {
	//open input file
	std::fstream fin;

	fin.open(filname);
	std::string line;
	//go through lines and get the input values, each line is a value vector
	while (std::getline(fin, line)) 
	{
		std::vector<float> data_line;
		//process exact number
		for (auto i = line.begin(); i != line.end(); i++) 
		{
			if (*i == '0' || *i == '1') { data_line.push_back(*i  - '0'); }
			else if (*i == '-') { data_line.push_back(-1.0); i++; }
		}
		//line is 1 vector
		data.push_back(new Matrix(data_line.size(),1));

		for (std::size_t i = 0; i < data_line.size(); i++) {data.back()->coeffRef(i,0) = data_line[i];}
	}
}

#endif
