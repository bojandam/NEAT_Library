#ifndef NEAT_LIBRARY_INDIVIDUAL_H
#define NEAT_LIBRARY_INDIVIDUAL_H
#include "genome.h"
namespace NEAT
{
    struct Individual
    {
        Genome genome;
        double fitness;
        Individual(Genome && genome, const double & fitness) :genome(genome), fitness(fitness) {}
        bool operator<(const Individual & other) {
            return this->fitness < other.fitness;
        }
    };


}

#endif