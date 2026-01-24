#ifndef NEAT_LIBRARY_NEAT_H
#define NEAT_LIBRARY_NEAT_H

#include "includes.h" 
#include "genome.h"
#include "phenotype.h"
#include "individual.h"
namespace NEAT {
    class NEAT {
    protected:
        std::uniform_real_distribution<double> weightDistribution; //range of values weights can be
        std::uniform_real_distribution<double> nodeDistribution; //range of values nodes can have
        std::random_device rd;
        std::mt19937 rnd;

        uint nodeIdCounter;
        uint innovationCounter = 0;

        // maps used to keep track of innovation on a gennerational level so duplicate innovations get the same innovation number
        std::map<Genome::Link, int> innovationTracker_AddedConnections;//[newLink]->newLinks inovation id
        std::map<Genome::Link, int> innovationTracker_AddedNode;// [link that got replased]-> Id of the new node 
        //ex: (1->4) :-> (1->6),(6->4):  AddedNode[(1->4)]=6,  AddedConnections[(1->6) or (6->4)] = it's innovation id

        uint numOfInputsInNN;
        uint numOfOutputsInNN;
        uint generationSize;
        std::uniform_int_distribution<uint> individualSelector;

    public:
        NEAT(uint numOfInputsInNN, uint numOfOutputsInNN, uint generationSize = 1000) :weightDistribution(-1, 1), nodeDistribution(0, 1),
            numOfInputsInNN(numOfInputsInNN), numOfOutputsInNN(numOfOutputsInNN), generationSize(generationSize),
            rd(), rnd(rd), individualSelector(0, generationSize) {
            nodeIdCounter = numOfInputsInNN + numOfOutputsInNN;
        }
        void  RunAlgorithm(double (*FitnessFunction)(const Phenotype &),
            bool (*TermanateCondition)(const std::vector<double> & GenerationFitness, uint IterationsFinished),
            double selectionRate,
            const std::vector<Genome::Link> & StarterLinks = {});

        Genome Crossover(const Individual & fitter, const Individual & lessFit);
        void Mutate(Genome & child);
    };





    void NEAT::RunAlgorithm(
        double (*FitnessFunction)(const Phenotype &),
        bool (*TermanateCondition)(const std::vector<double> & GenerationFitness, uint IterationsFinished),
        double selectionRate,
        const std::vector<Genome::Link> & StarterLinks = {})
    {
        //Initalization
        std::vector<Genome::Connection> StarterConnections;
        for (Genome::Link link : StarterLinks) {
            StarterConnections.push_back(Genome::Connection(link, weightDistribution(rnd), true, innovationCounter++));
        }
        std::vector<Genome> Generation(generationSize, Genome(numOfInputsInNN, numOfOutputsInNN, weightDistribution, rnd, StarterConnections));

        std::vector<Individual> GenerationIndividuals;
        std::vector<double> GenerationFitness;
        uint itterationsFinished = 0;
        do {
            //Calculate Fitness
            for (Genome & genome : Generation) {
                double genomeFitness = FitnessFunction(std::move(Phenotype(genome)));
                GenerationIndividuals.push_back(std::move(Individual(std::move(genome), genomeFitness)));
                GenerationFitness.push_back(genomeFitness);
            }//!!! Generation is unusable now

            //Selection
            std::sort(GenerationIndividuals.begin(), GenerationIndividuals.end());
            //Crossover & Mutation
            std::vector<Genome> NextGeneration;
            while (NextGeneration.size() < generationSize)
            {
                Individual * parentA = &GenerationIndividuals[individualSelector(rnd)];
                Individual * parentB = &GenerationIndividuals[individualSelector(rnd)];
                if (parentA != parentB) {//to do: Speculation
                    if (parentA->fitness < parentB->fitness)
                        std::swap(parentA, parentB);
                    Genome child = Crossover(*parentA, *parentB);
                    Mutate(child);
                    NextGeneration.push_back(std::move(child));
                }
            }
            Generation = std::move(NextGeneration);
            itterationsFinished++;
        } while (!TermanateCondition(GenerationFitness, itterationsFinished));
        //Termanation

    }
}
#endif  