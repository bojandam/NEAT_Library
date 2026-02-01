#ifndef NEAT_LIBRARY_INCLUDES_H
#define NEAT_LIBRARY_INCLUDES_H
#include<vector>
#include<map>
#include<cmath>
#include<random>
#include<algorithm>
#include<stack>

namespace bNEAT {
    namespace StandardActivationFunctions {
        double Sigmoid(double x);
        double NEATSigmoid(double x);
    }
    typedef double (*ActivationFunction) (double);
    typedef unsigned int uint;
}


#endif