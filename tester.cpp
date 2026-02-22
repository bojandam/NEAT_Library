#include <iostream>
#include "NEAT Implementation\NEAT.hpp"
#include <algorithm> 
#include <iomanip>
using namespace std;

template<typename T> T sqr(T x) { return x * x; }

double FitnessSqr(const bNEAT::Phenotype & agent) {
    static std::normal_distribution<double> dist(0, 5.0);
    static std::mt19937 rng(std::random_device{}());
    // cout << "-------\n";
    double rez = 0;
    for (int i = 0; i < 10; i++) {
        double x = dist(rng);
        double prediction = agent.Predict({ x })[0];

        // std::cout << x << ": " << prediction << std::endl;
        rez += -sqr(x * x - prediction);
    }
    // cout << "-------\n";
    return rez;
}
double FitnessXOR(const bNEAT::Phenotype & agent) {
    double rez = 0;
    for (double i = 0; i < 2; i++) {
        for (double j = 0; j < 2; j++) {
            double pred = agent.Predict({ i,j })[0];
            // cout << setw(20) << "[" + to_string((int)i) + "," + to_string((int)j) + "] -> [" + to_string(pred) + "] {" + to_string(((int)i ^ (int)j)) + "}  |   ";
            rez += abs(((int)((int)i ^ (int)j)) - pred);
        }
    }
    rez = 4 - rez;
    rez = rez * rez;
    // cout << rez << endl;
    return rez;
}


bool TerminateXOR(std::vector<double>  Fitness, unsigned int itterations) {
    sort(Fitness.begin(), Fitness.end(), [](const double & a, const double & b) {return a > b; });
    if (itterations % 10 == 0)
        cout << itterations << ": " << Fitness[0] << endl;
    return itterations >= 10000 || Fitness[10] >= 15.999;
}
bool TerminateSqr(std::vector<double>  Fitness, unsigned int itterations) {
    sort(Fitness.begin(), Fitness.end(), [](const double & a, const double & b) {return a > b; });
    if (itterations % 100 == 0)
        cout << itterations << ": " << Fitness[0] << endl;
    return itterations >= 100000 || Fitness[10] >= -0.5;
}

int main()
{
    using namespace bNEAT;
    // while (true)
    // {
    //     double x = 1, prediction;
    //     cin >> prediction;
    //     cout << 1 / (0.000001 + std::abs(x * x - prediction)) << endl;
    // }

    NEAT algorithm(1, 1, 150);
    std::vector<Individual> rez = algorithm.RunAlgorithm(FitnessSqr, TerminateSqr);
    Phenotype agent(rez[0].genome, algorithm.activationFunction);
    cout << "---------------\n";
    // for (int a = 0; a < 2; a++)
    //     for (int b = 0; b < 2; b++)
    //         cout << fixed << a << "^" << b << " = " << (a ^ b) << " --> " << agent.Predict({ (double)a, (double)b })[0] << endl;
    while (true)
    {
        double x; cin >> x;
        cout << agent.Predict({ x })[0] << endl;

    }


    return 0;
}