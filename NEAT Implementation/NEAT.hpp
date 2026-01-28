#ifndef NEAT_LIBRARY_NEAT_H
#define NEAT_LIBRARY_NEAT_H

#include "includes.hpp" 
#include "genome.hpp"
#include "phenotype.hpp"
#include "individual.hpp"
#include "species.hpp"
namespace NEAT {

    class NEAT {
    private:
        template<typename T> T choose_between(T optionA, T optionB);
        template<typename T> void CrossoverVector(const std::vector<T> & fitter, const std::vector<T> & lessFit, std::vector<T> & childVector, uint(*getId)(const T & v));
    protected:
        std::uniform_real_distribution<double> weightDistribution; //range of values weights can be
        std::uniform_real_distribution<double> nodeDistribution; //range of values nodes can have
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

        //Compatibility calculation Coefficients
        double compatibilityExcessCoefficient = 1.0;
        double compatibilityDisjointCoefficient = 1.0;
        double compatibilityMatchingCoefficient = 0.4;
        uint compatibilityMinNThreshold = 20;
        double compatibilityTreshold = 3.0;

        //
        double weightMutationPerWeightProbability = 0.03;
        double biasMutationPerNodeProbability = 0.03;
        double addNodeMutationProbability = 0.1;
        double addConnectionMutationProbability = 0.1;
        std::normal_distribution<double> deltaNudgeDistribution;

        double CalculateCompatibility(const Genome & A, const Genome & B);
        void MutateWeight(double & w);
        void MutateAddNode(Genome & child);
        void MutateAddConnection(Genome & child);
    public:
        NEAT(uint numOfInputsInNN, uint numOfOutputsInNN, uint generationSize = 1000) :
            weightDistribution(-1, 1), nodeDistribution(0, 1), deltaNudgeDistribution(),
            numOfInputsInNN(numOfInputsInNN), numOfOutputsInNN(numOfOutputsInNN), generationSize(generationSize),
            rnd(std::random_device{}()), individualSelector(0, generationSize - 1) {
            nodeIdCounter = numOfInputsInNN + numOfOutputsInNN;
        }
        std::vector<Genome>  RunAlgorithm(double (*FitnessFunction)(const Phenotype &),
            bool (*TermanateCondition)(const std::vector<double> & GenerationFitness, uint IterationsFinished),
            double selectionRate,
            const std::vector<Genome::Link> & StarterLinks = {});

        Genome Crossover(const Individual & fitter, const Individual & lessFit);
        void Mutate(Genome & child);
    };





    std::vector<Genome> NEAT::RunAlgorithm(
        double (*FitnessFunction)(const Phenotype &),
        bool (*TermanateCondition)(const std::vector<double> & GenerationFitness, uint IterationsFinished),
        double selectionRate,
        const std::vector<Genome::Link> & StarterLinks)
    {
        //1.Initalization
        //Starter Links -> Starter Connections
        std::vector<Genome::Connection> StarterConnections;
        for (Genome::Link link : StarterLinks)
            StarterConnections.push_back(Genome::Connection(link, weightDistribution(rnd), true, innovationCounter++));

        std::vector<Genome> Generation(generationSize, Genome(numOfInputsInNN, numOfOutputsInNN, weightDistribution, rnd, StarterConnections));
        std::vector<Species> SpeciesTracker;
        std::vector<double> GenerationFitness;

        uint itterationsFinished = 0;
        do {
            GenerationFitness.clear();
            //2.Calculate Fitness and Speciate
            std::vector<Individual * > GenerationIndividuals;
            for (Genome & genome : Generation) {
                double genomeFitness = FitnessFunction(std::move(Phenotype(genome)));
                Individual individual(std::move(genome), genomeFitness);

                bool added = false;
                for (Species & specie : SpeciesTracker) {
                    if (CalculateCompatibility(individual.genome, specie.representitive.genome) < compatibilityTreshold) {
                        specie.members.push_back(std::move(individual));
                        added = true;
                        break;
                    }
                }
                if (!added) {
                    SpeciesTracker.push_back(Species(std::move(individual)));
                }
                GenerationFitness.push_back(genomeFitness);
            }//!!! Generation is unusable now
            //3.Selection
            for (Species & spicie : SpeciesTracker) {
                for (Individual & individual : spicie.members) {
                    individual.fitness /= spicie.members.size(); //updated fitnes using speciation
                    GenerationIndividuals.push_back(&individual);
                }
            }
            std::sort(GenerationIndividuals.begin(), GenerationIndividuals.end(), [](Individual * a, Individual * b) {return *a < *b; });
            //Crossover & Mutation
            std::vector<Genome> NextGeneration;
            while (NextGeneration.size() < generationSize)
            {
                Individual * parentA = GenerationIndividuals[individualSelector(rnd)];
                Individual * parentB = GenerationIndividuals[individualSelector(rnd)];
                if (parentA != parentB) {
                    if (parentA->fitness < parentB->fitness)
                        std::swap(parentA, parentB);
                    Genome child = Crossover(*parentA, *parentB);
                    Mutate(child);
                    NextGeneration.push_back(std::move(child));
                }
            }
            Generation = std::move(NextGeneration);
            for (Species & specie : SpeciesTracker)
                specie.PrepareForNextGeneration();
            itterationsFinished++;
        } while (!TermanateCondition(GenerationFitness, itterationsFinished));
        //Termanation
        return Generation;
    }


    double NEAT::CalculateCompatibility(const Genome & A, const Genome & B) {
        uint N = std::max(A.connections.size(), B.connections.size());
        N = (N < compatibilityMinNThreshold) ? 1 : N;
        double Excess = 0;
        double Disjoint = 0;
        double WeightDiff = 0;
        uint Matching = 0;
        typedef std::vector<Genome::Connection>::const_iterator IT;
        IT i = A.connections.begin(), j = B.connections.begin();
        while (i != A.connections.end() && j != B.connections.end()) {
            if (i->innovationNumber == j->innovationNumber)
            {
                WeightDiff += std::abs(i->weight - j->weight);
                Matching++;
                i++;
                j++;
            }
            else {
                IT & bigger = (i->innovationNumber > j->innovationNumber) ? i : j;
                IT & smaller = (i == bigger) ? j : i;
                Disjoint++;
                smaller++;
            }
        }
        Excess = std::distance(i, A.connections.end()) + std::distance(j, B.connections.end());
        return compatibilityDisjointCoefficient * (Disjoint / N)
            + compatibilityExcessCoefficient * (Excess / N)
            + compatibilityMatchingCoefficient * (Matching ? WeightDiff / Matching : 0.0);
    }

    template<typename T> T NEAT::choose_between(T A, T B) {
        static std::uniform_int_distribution<int> YesNoRange(0, 1);
        return (YesNoRange(rnd) ? A : B);
    }

    template<typename T> void NEAT::CrossoverVector(const std::vector<T> & fitter, const std::vector<T> & lessFit, std::vector<T> & childVector, uint(*getId)(const T & v)) {
        typename std::vector<T>::const_iterator i = fitter.begin(), j = lessFit.begin();
        while (i != fitter.end() && j != lessFit.end()) {
            if (getId(*i) == getId(*j)) {
                childVector.connections.push_back(choose_between(*i, *j));
                i++; j++;
            }
            else if (getId(*i) > getId(*j)) {
                j++;
            }
            else {
                childVector.connections.push_back(*i);
                i++;
            }
        }
    }

    Genome NEAT::Crossover(const Individual & fitter, const Individual & lessFit) {
        Genome child(numOfInputsInNN, numOfOutputsInNN, weightDistribution, rnd);

        CrossoverVector<Genome::Connection>(
            fitter.genome.connections,
            lessFit.genome.connections,
            child.connections,
            [](const Genome::Connection & v) {return v.innovationNumber; }
        );

        CrossoverVector<Genome::Node>(
            fitter.genome.nodes,
            lessFit.genome.nodes,
            child.nodes,
            [](const Genome::Node & v) {return v.id; }
        );
        return child;
    }
}
#endif  