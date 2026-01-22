#ifndef NEAT_LIBRARY_NEAT_H
#define NEAT_LIBRARY_NEAT_H

#include "includes.h" 
#include "genome.h"
#include "phenotype.h"

namespace NEAT {
    class NEAT {
    protected:
        std::uniform_real_distribution<double> weightDistribution; //range of values weights can be
        std::uniform_real_distribution<double> nodeDistribution; //range of values nodes can have
        std::random_device rnd;

        uint nodeIdCounter;
        uint innovationCounter = 0;

        // maps used to keep track of innovation on a gennerational level so duplicate innovations get the same innovation number
        std::map<Genome::Link, int> innovationTracker_AddedConnections;//[newLink]->newLinks inovation id
        std::map<Genome::Link, int> innovationTracker_AddedNode;// [link that got replased]-> Id of the new node 
        //ex: (1->4) :-> (1->6),(6->4):  AddedNode[(1->4)]=6,  AddedConnections[(1->6) or (6->4)] = it's innovation id

        uint numOfInputsInNN;
        uint numOfOutputsInNN;
        uint generationSize;

    public:
        NEAT(uint numOfInputsInNN, uint numOfOutputsInNN, uint generationSize = 1000) :weightDistribution(-1, 1), nodeDistribution(0, 1),
            numOfInputsInNN(numOfInputsInNN), numOfOutputsInNN(numOfOutputsInNN), generationSize(generationSize) {
            nodeIdCounter = numOfInputsInNN + numOfOutputsInNN;
        }
        void  RunAlgorithm(double (*FitnessFunction)(const Phenotype &),
            bool (*TermanateCondition)(const std::vector<double> & GenerationFitness, uint IterationsFinished),
            const std::vector< Genome::Link> & StarterLinks = {});
    };





    void NEAT::RunAlgorithm(double (*FitnessFunction)(const Phenotype &),
        bool (*TermanateCondition)(const std::vector<double> & GenerationFitness, uint IterationsFinished),
        const std::vector<Genome::Link> & StarterLinks = {})
    {
        //Initalization
        std::vector<Genome::Connection> StarterConnections;
        for (Genome::Link link : StarterLinks) {
            StarterConnections.push_back(Genome::Connection(link, weightDistribution(rnd), true, innovationCounter++));
        }
        std::vector<Genome> Generation(generationSize, Genome(numOfInputsInNN, numOfOutputsInNN, weightDistribution, rnd, StarterConnections));

        std::vector<double> GenerationFitness(generationSize);
        uint itterationsFinished = 0;
        do {
            //Calculate Fitness
            std::vector<Phenotype> Phenotypes;
            for (const Genome & genome : Generation) {
                Phenotypes.push_back(Phenotype(genome));
            }
            for (const Phenotype & phenotype : Phenotypes) {
                GenerationFitness.push_back(FitnessFunction(phenotype));
            }

            //Selection
                //How do we make this customisable

            //Crossover

            //Mutation



            itterationsFinished++;
        } while (!TermanateCondition(GenerationFitness, itterationsFinished));
        //Termanation

    }
}
#endif  