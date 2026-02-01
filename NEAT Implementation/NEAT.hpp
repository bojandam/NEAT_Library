#ifndef NEAT_LIBRARY_NEAT_H
#define NEAT_LIBRARY_NEAT_H

#include "includes.hpp" 
#include "genome.hpp"
#include "phenotype.hpp"
#include "individual.hpp"
#include "species.hpp"
#include <list>
namespace bNEAT {

    class NEAT {
    private:
        template<typename T> T choose_between(T optionA, T optionB);
        template<typename T> T & choose_within(std::vector<T> & options);
        template<typename T> void CrossoverVector(const std::vector<T> & fitter, const std::vector<T> & lessFit, std::vector<T> & childVector, uint(*getId)(const T & v));
        std::uniform_real_distribution<double> decider;
        void MarkForDeath(std::vector<Species> & SpeciesTracker);
    public:
        std::normal_distribution<double> weightDistribution; //range of values weights can be
        std::mt19937 rnd;

        uint nodeIdCounter;
        uint innovationCounter = 0;

        // maps used to keep track of innovation on a gennerational level so duplicate innovations get the same innovation number
        std::map<Genome::Link, uint> innovationTracker_AddedConnections;//[newLink]->newLinks inovation id
        std::map<Genome::Link, uint> innovationTracker_AddedNode;// [link that got replased]-> Id of the new node 
        //ex: (1->4) :-> (1->6),(6->4):  AddedNode[(1->4)]=6,  AddedConnections[(1->6) or (6->4)] = it's innovation id

        uint numOfInputsInNN;
        uint numOfOutputsInNN;
        uint generationSize;
        double selectionRate = 0.4;
        std::uniform_int_distribution<uint> reproductionSelector;

        ActivationFunction activationFunction = StandardActivationFunctions::NEATSigmoid;

        //Compatibility calculation Coefficients
        double compatibilityExcessCoefficient = 1.0;
        double compatibilityDisjointCoefficient = 1.0;
        double compatibilityMatchingCoefficient = 0.4;
        uint compatibilityMinNThreshold = 20;
        double compatibilityTreshold = 2.0;

        //Crossover Coefficients
        double percentOf_MutateOnly_Children = 0.25;
        double interSpicie_Crossover_Chance = 0.01;

        //Mutation Coefficients        
        double weightMutation_PerWeight_Probability = 0.8;
        double biasMutation_PerNode_Probability = 0.8;
        double addNodeMutation_Probability = 0.2;
        double addConnectionMutation_Probability = 0.4;
        double mutationRandomToNudgeRatio = 0.1;
        double mutationNudgeCap = 0.25;
        std::normal_distribution<double> deltaNudgeDistribution;

        double CalculateCompatibility(const Genome & A, const Genome & B);
        void MutateWeightNudge(double & w);
        void MutateWeightRandom(double & w);
        void MutateAddNode(Genome & child);
        void MutateAddConnection(Genome & child);
        Genome Crossover(const Individual & fitter, const Individual & lessFit);
        void Mutate(Genome & child);

        typedef std::vector<Species> SpeciesTracker;
        void RemoveExtinctSpecies(SpeciesTracker & speciesTracker);
        void RemoveStagnantSpecies(SpeciesTracker & speciesTracler);
        void Speciate(SpeciesTracker & speciesTracker, Individual && individual);
    public:
        NEAT(uint numOfInputsInNN, uint numOfOutputsInNN, uint generationSize = 1000) :
            weightDistribution(0, 1.0), deltaNudgeDistribution(0, 0.1),
            numOfInputsInNN(numOfInputsInNN), numOfOutputsInNN(numOfOutputsInNN), generationSize(generationSize),
            rnd(std::random_device{}()),
            reproductionSelector(0, (int)((generationSize - 1) * selectionRate)),
            decider(0, 1)
        {
            nodeIdCounter = numOfInputsInNN + numOfOutputsInNN;
        }
        std::vector<Individual>  RunAlgorithm(double (*FitnessFunction)(const Phenotype &),
            bool (*TermanateCondition)(std::vector<double>   GenerationFitness, uint IterationsFinished),
            const std::vector<Genome::Link> & StarterLinks = {});


    };
}

#endif  