#ifndef NEAT_LIBRARY_PHENOTYPE_H
#define NEAT_LIBRARY_PHENOTYPE_H
#include"genome.hpp"

namespace NEAT
{
    struct Phenotype
    {
        Phenotype(const Genome &);
        std::vector<double> Predict(std::vector<double> input) const;
    };
}


#endif