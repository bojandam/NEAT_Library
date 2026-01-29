#ifndef NEAT_LIBRARY_SPECIES_H
#define NEAT_LIBRARY_SPECIES_H

#include"individual.hpp"
namespace bNEAT {
    struct Species
    {
        Individual representitive;
        std::vector<Individual> members;
        int lifetime = 0;
        void PrepareForNextGeneration(int newRepresentativeIndex = 0) {
            representitive = std::move(members[newRepresentativeIndex]);
            members.clear();
        }
        Species(Individual && representitive) :representitive(representitive) {
            members.push_back(this->representitive);
            lifetime++;
        }
    };
}



#endif