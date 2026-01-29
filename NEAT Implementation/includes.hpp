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
        double Sigmoid(double x) {
            return 1 / (1 + std::exp(-x));
        }
    }
    typedef double (*ActivationFunction) (double);
    typedef unsigned int uint;
}


#endif