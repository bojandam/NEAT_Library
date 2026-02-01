#ifndef NEAT_LIBRARY_SPECIES_H
#define NEAT_LIBRARY_SPECIES_H

#include"individual.hpp"
namespace bNEAT {
    struct Species
    {
        Individual representitive;
        std::vector<Individual> members;
        int lifetime = 0;
        double maxFitness = 0;
        double prevMaxFitness = 0;
        int lifetime_When_MaxFitness_Changed = 0;

        void PrepareForNextGeneration(int newRepresentativeIndex = 0) {
            if (!members.empty()) {
                representitive = std::move(members[newRepresentativeIndex]);
                members.clear();
                lifetime++;
            }
        }
        void KillMarkedSpecies() {
            members.erase(
                std::find_if(members.begin(), members.end(), [](const Individual & x) {return x.markedForDeath; }),//members is sorted by this
                members.end()
            );
        }
        Species(Individual && representitive) :representitive(representitive) {
            members.push_back(this->representitive);
            lifetime++;
        }
    };
}



#endif