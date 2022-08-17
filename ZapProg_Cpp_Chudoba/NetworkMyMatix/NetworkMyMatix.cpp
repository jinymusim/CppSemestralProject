// NetworkMyMatix.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "NeuralNetwork.hpp"
#include "loadCSV.hpp"

using t_args = std::vector<std::string>;

//simple int parser
void get_network_params(const std::string& params, std::vector<std::size_t>& params_vector) {

    int temp = 0;
    //goes through given string and parsers digits
    for (std::size_t i = 0; i < params.size(); i++)
    {
        //if separation symbol is found push back the number
        if (params[i] == ',' || params[i] == ';')
        {
            params_vector.push_back(temp);
            temp = 0;
        }
        //if not recalculate the number
        else
        {
            temp = temp * 10 + (params[i] - '0');
        }
    }
    //if there is no separation at the end, but there still is processed number save it
    if (temp != 0) { params_vector.push_back(temp); }
}

//simple argument parse for type of network creation
NeuralNetwork create_network_form_args(const std::vector<std::string>& args) {
    //always from the first argument
    auto&& arg = args[0];

    //based on argument create network
    if (arg[0] == '-') 
    {
        auto&& argChar = arg[1];
        switch (argChar)
        {
        case 'n': {
            std::vector<std::size_t> networkParams;
            get_network_params(args[1], networkParams);
            return NeuralNetwork(networkParams);
        }
        case 'r':
            return NeuralNetwork(args[1]);
        default:
            return NeuralNetwork();
        }
    }
}


int main(int argc, char** argv)
{   //Nakonec 257 vstup 7 vystup.
    //Problem s ridkosti dat, nemuze dobre prepocitat weights, vyreseno ze misto 0 davame -1
    t_args arguments(argv + 1, argv + argc);
    if (arguments.size() < 4) {
        std::cerr << "Bad amount of arguments" << std::endl;
        return 1;
    }
    NeuralNetwork network = create_network_form_args(arguments);


    //network_1_hidden.txt for network with 1 hidden layer, network_2_hidden.txt for network with 2 hidden layers
    //1 hidden works better, maybe not enough data for 2 hidden layers, or low amount of training (still 30 more than 1 hidden)


    std::vector<Matrix*> testIn;
    load_data(arguments[2], testIn);
    std::vector<Matrix*> testOut;
    load_data(arguments[3], testOut);

    //if file name was incorrect, or the file is elsewhere end the program, so no error can occur
    if (testIn.size() == 0 || testOut.size() == 0) {
        std::cerr << "Couldn't parse file" << std::endl;
        return 1;
    }

    //based on the fact, if the argument is present train that number of generation
    std::vector<std::size_t> repetition;
    if (arguments.size() > 4) { get_network_params(arguments[4], repetition);}
    else{ 
        std::cerr << "Number of generations is not present" << std::endl;
        repetition.push_back(1);
    }

    for (int i = 0; i < repetition.back(); i++) {
        network.train(testIn, testOut);
    }


    //network saving, if filename is present
    if (arguments.size() > 5) { network.saveLearnedNetwork(arguments[5]); }
    else {
        std::cerr << "filename to save network not present" << std::endl;
        return 1;
    }
    


}
