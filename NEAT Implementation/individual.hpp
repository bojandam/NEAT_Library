#ifndef NEAT_LIBRARY_INDIVIDUAL_H
#define NEAT_LIBRARY_INDIVIDUAL_H
#include "genome.hpp"
namespace bNEAT
{
    struct Individual
    {
        Genome genome;
        double fitness;
        Individual(Genome && genome, const double & fitness) :genome(std::move(genome)), fitness(fitness) {}
        bool operator<(const Individual & other) {
            return this->fitness < other.fitness;
        }
    };


}

#endif