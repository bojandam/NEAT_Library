#include "includes.hpp"

namespace bNEAT {
    namespace StandardActivationFunctions {
        double Sigmoid(double x) {
            return 1 / (1 + std::exp(-x));
        }
        double NEATSigmoid(double x) {
            return 1 / (1 + std::exp(-4.9 * x));
        }
    }
}