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
        template<typename T> T & choose_within(std::vector<T> options);

        template<typename T> void CrossoverVector(const std::vector<T> & fitter, const std::vector<T> & lessFit, std::vector<T> & childVector, uint(*getId)(const T & v));
    protected:
        std::uniform_real_distribution<double> weightDistribution; //range of values weights can be
        std::uniform_real_distribution<double> nodeDistribution; //range of values nodes can have (what?)
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
        ActivationFunction activationFunction = StandardActivationFunctions::Sigmoid;

        //Compatibility calculation Coefficients
        double compatibilityExcessCoefficient = 1.0;
        double compatibilityDisjointCoefficient = 1.0;
        double compatibilityMatchingCoefficient = 0.4;
        uint compatibilityMinNThreshold = 20;
        double compatibilityTreshold = 3.0;

        //Mutation Coefficients
        double weightMutation_PerWeight_Probability = 0.03;
        double biasMutation_PerNode_Probability = 0.03;
        double addNodeMutation_Probability = 0.1;
        double addConnectionMutation_Probability = 0.1;
        double mutationRandomToNudgeRatio = 0.5;
        double mutationNudgeCap = 0.25;
        std::normal_distribution<double> deltaNudgeDistribution;

        double CalculateCompatibility(const Genome & A, const Genome & B);
        void MutateWeightNudge(double & w);
        void MutateWeightRandom(double & w);
        void MutateAddNode(Genome & child);
        void MutateAddConnection(Genome & child);
        Genome Crossover(const Individual & fitter, const Individual & lessFit);
        void Mutate(Genome & child);
    public:
        NEAT(uint numOfInputsInNN, uint numOfOutputsInNN, uint generationSize = 1000) :
            weightDistribution(-1, 1), nodeDistribution(0, 1), deltaNudgeDistribution(),
            numOfInputsInNN(numOfInputsInNN), numOfOutputsInNN(numOfOutputsInNN), generationSize(generationSize),
            rnd(std::random_device{}()), individualSelector(0, generationSize - 1)
        {
            nodeIdCounter = numOfInputsInNN + numOfOutputsInNN;
        }
        std::vector<Genome>  RunAlgorithm(double (*FitnessFunction)(const Phenotype &),
            bool (*TermanateCondition)(const std::vector<double> & GenerationFitness, uint IterationsFinished),
            double selectionRate,
            const std::vector<Genome::Link> & StarterLinks = {});


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
        //utils definition
        std::vector<Genome> Generation(generationSize, Genome(numOfInputsInNN, numOfOutputsInNN, weightDistribution, rnd, StarterConnections));
        std::vector<Species> SpeciesTracker;
        std::vector<double> GenerationFitness;

        uint itterationsFinished = 0;
        do {
            //prepare utils
            GenerationFitness.clear();
            innovationTracker_AddedConnections.clear();
            innovationTracker_AddedNode.clear();
            //2.Calculate Fitness and Speciate
            for (Genome & genome : Generation)
            {
                double genomeFitness = FitnessFunction(Phenotype(genome, activationFunction));  // calculate Fitness
                Individual individual(std::move(genome), genomeFitness);    // genome -> individual 
                //Speciation
                bool joinedSpecie = false;
                for (Species & specie : SpeciesTracker) {
                    if (CalculateCompatibility(individual.genome, specie.representitive.genome) < compatibilityTreshold) // check if individual fits in specie
                    {
                        specie.members.push_back(std::move(individual)); // individuals get moved to spiciation tracker (/table/organiser) 
                        joinedSpecie = true;
                        break;
                    }
                }
                if (!joinedSpecie) {
                    SpeciesTracker.push_back(Species(std::move(individual)));//make a new specie
                }
                GenerationFitness.push_back(genomeFitness);
            }
            Generation.clear();// just in case (since all genomes are now in Spicies tracker)
            //3.Selection
            std::vector<Individual * > GenerationIndividuals;
            for (Species & spicie : SpeciesTracker) {
                for (Individual & individual : spicie.members) {
                    individual.fitness /= spicie.members.size(); //updated fitnes using speciation
                    GenerationIndividuals.push_back(&individual);
                }
            }
            std::sort(GenerationIndividuals.begin(), GenerationIndividuals.end(), [](Individual * a, Individual * b) {return *a < *b; });
            //4.Crossover & Mutation
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
            //Prepare for next Loop
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
    template<typename T> T & NEAT::choose_within(std::vector<T> options) {
        std::uniform_int_distribution<uint> SelectionRange(0, options.size());
        return options[SelectionRange(rnd)];
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

        CrossoverVector<Node>(
            fitter.genome.nodes,
            lessFit.genome.nodes,
            child.nodes,
            [](const Node & v) {return v.id; }
        );
        return child;
    }

    void NEAT::MutateAddNode(Genome & child) {
        if (child.connections.empty())
            return;

        Genome::Connection & oldConnection = choose_within(child.connections);
        const Genome::Link & oldLink = oldConnection.link;

        auto it = innovationTracker_AddedNode.find(oldLink);
        int NodeId = (it == innovationTracker_AddedNode.end() ? (innovationTracker_AddedNode[oldLink] = nodeIdCounter++) : it->second);

        Genome::Link linkTo(oldLink.nodeInId, NodeId), linkFrom(NodeId, oldLink.nodeOutId);

        if (it == innovationTracker_AddedNode.end()) {
            innovationTracker_AddedConnections[linkTo] = innovationCounter++;
            innovationTracker_AddedConnections[linkFrom] = innovationCounter++;
        }
        Genome::Connection new_connection_To(linkTo, 1.0, true, innovationTracker_AddedConnections[linkTo]);
        Genome::Connection new_connection_From(linkFrom, oldConnection.weight, true, innovationTracker_AddedConnections[linkFrom]);

        oldConnection.isEnabled = false;
        child.nodes.push_back(Node(NodeId, weightDistribution(rnd)));
        child.connections.push_back(new_connection_From);
        child.connections.push_back(new_connection_To);
    }

    void NEAT::MutateAddConnection(Genome & child) {
        Node & fromNode = choose_within(child.nodes);
        Node & toNode = choose_within(child.nodes);
        Genome::Link newLink(fromNode.id, toNode.id);
        if (child.link_would_create_loop(newLink))
            return;
        if (innovationTracker_AddedConnections.find(newLink) == innovationTracker_AddedConnections.end())
            innovationTracker_AddedConnections[newLink] = innovationCounter++;
        child.connections.push_back({ newLink, weightDistribution(rnd), true, innovationTracker_AddedConnections[newLink] });
    }

    void NEAT::MutateWeightNudge(double & w)
    {
        w = std::clamp<double>(w + std::clamp<double>(deltaNudgeDistribution(rnd), -mutationNudgeCap, mutationNudgeCap), weightDistribution.min(), weightDistribution.max());
    }

    void NEAT::MutateWeightRandom(double & w)
    {
        w = weightDistribution(rnd);
    }

    void NEAT::Mutate(Genome & child)
    {
        static std::uniform_real_distribution<double> decider(0, 1);

        if (decider(rnd) <= addConnectionMutation_Probability)
            MutateAddConnection(child);
        if (decider(rnd) <= addNodeMutation_Probability)
            MutateAddConnection(child);
        for (Node & node : child.nodes) {
            if (decider(rnd) <= biasMutation_PerNode_Probability)
                (decider(rnd) <= mutationRandomToNudgeRatio) ? MutateWeightRandom(node.bias) : MutateWeightNudge(node.bias);
        }
        for (Genome::Connection & connection : child.connections) {
            if (decider(rnd) <= weightMutation_PerWeight_Probability)
                (decider(rnd) <= mutationRandomToNudgeRatio) ? MutateWeightRandom(connection.weight) : MutateWeightNudge(connection.weight);
        }

    }

}
#endif  