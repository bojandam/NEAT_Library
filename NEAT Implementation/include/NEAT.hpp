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
        std::vector<Genome>  RunAlgorithm(double (*FitnessFunction)(const Phenotype &),
            bool (*TermanateCondition)(std::vector<double>   GenerationFitness, uint IterationsFinished),
            const std::vector<Genome::Link> & StarterLinks = {});


    };





    std::vector<Genome> NEAT::RunAlgorithm(
        double (*FitnessFunction)(const Phenotype &),
        bool (*TermanateCondition)(std::vector<double> GenerationFitness, uint IterationsFinished),
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
            //3.Speciation

            for (Species & spicie : SpeciesTracker) {
                for (Individual & individual : spicie.members) {
                    spicie.maxFitness = std::max(spicie.maxFitness, individual.fitness);
                    individual.fitness /= spicie.members.size(); //updated fitnes using speciation
                }
                if (spicie.maxFitness != spicie.prevMaxFitness) {
                    spicie.prevMaxFitness = spicie.maxFitness;
                    spicie.lifetime_When_MaxFitness_Changed = spicie.lifetime;
                }
            }
            //4.Selection
            std::vector<Genome> NextGeneration;

            auto it = std::stable_partition(
                SpeciesTracker.begin(),
                SpeciesTracker.end(),
                [](const Species & x) {return x.lifetime - x.lifetime_When_MaxFitness_Changed >= 20; }
            );
            if (it != SpeciesTracker.begin() && it != SpeciesTracker.begin() + 1) {
                SpeciesTracker.erase(it, SpeciesTracker.end());
            }

            MarkForDeath(SpeciesTracker);

            for (Species & specie : SpeciesTracker) {
                std::sort(specie.members.begin(), specie.members.end(), [](const Individual & a, const Individual & b) {return b < a; });
                if (specie.members.size() > 5) {
                    NextGeneration.push_back(specie.members[0].genome);
                }
                specie.members.erase(
                    std::find_if(specie.members.begin(), specie.members.end(), [](const Individual & x) {return x.markedForDeath; }),
                    specie.members.end()
                );
            }
            SpeciesTracker.erase(
                std::remove_if(SpeciesTracker.begin(), SpeciesTracker.end(), [](const Species & x) {return x.members.empty(); }),
                SpeciesTracker.end());

            //5.Crossover & Mutation
            while (NextGeneration.size() < generationSize * percentOf_MutateOnly_Children)
            {
                const Individual & parent = choose_within(choose_within(SpeciesTracker).members);
                Genome child(parent.genome);
                Mutate(child);
                NextGeneration.push_back(std::move(child));
            }

            while (NextGeneration.size() < generationSize)  //to do: make this seklect spicies
            {
                Species & specie = choose_within(SpeciesTracker);
                Individual * parentA = &choose_within(specie.members);
                Individual * parentB = (decider(rnd) <= interSpicie_Crossover_Chance) ? &choose_within(choose_within(SpeciesTracker).members) : &choose_within(specie.members);
                if (parentA != parentB) {
                    if (parentA->fitness < parentB->fitness)
                        std::swap(parentA, parentB);
                    Genome child = Crossover(*parentA, *parentB);
                    Mutate(child);
                    NextGeneration.push_back(std::move(child));
                }
            }
            //6.Prepare for next Loop
            Generation = std::move(NextGeneration);
            for (Species & specie : SpeciesTracker)
                specie.PrepareForNextGeneration();
            itterationsFinished++;
        } while (!TermanateCondition(std::move(GenerationFitness), itterationsFinished));
        //7.Termanation
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
    template<typename T> T & NEAT::choose_within(std::vector<T> & options) {
        std::uniform_int_distribution<uint> SelectionRange(0, options.size() - 1);
        return options[SelectionRange(rnd)];
    }
    template<typename T> void NEAT::CrossoverVector(
        const std::vector<T> & fitter,
        const std::vector<T> & lessFit,
        std::vector<T> & childVector,
        uint(*getId)(const T & v))
    {
        typename std::vector<T>::const_iterator i = fitter.begin(), j = lessFit.begin();
        while (i != fitter.end() && j != lessFit.end()) {
            if (getId(*i) == getId(*j)) {
                childVector.push_back(choose_between(*i, *j));
                i++; j++;
            }
            else if (getId(*i) > getId(*j)) {
                j++;
            }
            else {
                childVector.push_back(*i);
                i++;
            }
        }
    }
    void NEAT::MarkForDeath(std::vector<Species> & SpeciesTracker) {
        std::vector<Individual * > GenerationIndividuals;
        for (Species & spicie : SpeciesTracker) {
            for (Individual & individual : spicie.members) {
                GenerationIndividuals.push_back(&individual);
            }
        }
        std::sort(GenerationIndividuals.begin(), GenerationIndividuals.end(), [](Individual * a, Individual * b) {return *b < *a; });
        for (int i = GenerationIndividuals.size() * (1 - selectionRate); i < GenerationIndividuals.size(); i++)
            GenerationIndividuals[i]->markedForDeath = true;
    }
    Genome NEAT::Crossover(const Individual & fitter, const Individual & lessFit) {
        Genome child;

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
        child.connections.push_back(new_connection_To);//order matters
        child.connections.push_back(new_connection_From);
    }

    void NEAT::MutateAddConnection(Genome & child) {
        Node & fromNode = choose_within(child.nodes);
        Node & toNode = choose_within(child.nodes);
        Genome::Link newLink(fromNode.id, toNode.id);
        if (std::find_if(child.connections.begin(), child.connections.end(), [newLink](const Genome::Connection & conn) {return conn.link == newLink; }) != child.connections.end())
            return;
        if (child.link_would_create_loop(newLink))
            return;
        if (innovationTracker_AddedConnections.find(newLink) == innovationTracker_AddedConnections.end())
            innovationTracker_AddedConnections[newLink] = innovationCounter++;
        child.connections.push_back({ newLink, weightDistribution(rnd), true, innovationTracker_AddedConnections[newLink] });
    }

    void NEAT::MutateWeightNudge(double & w)
    {
        w += std::clamp<double>(deltaNudgeDistribution(rnd), -mutationNudgeCap, mutationNudgeCap);
    }

    void NEAT::MutateWeightRandom(double & w)
    {
        w = weightDistribution(rnd);
    }

    void NEAT::Mutate(Genome & child) {
        if (decider(rnd) <= addConnectionMutation_Probability)
            MutateAddConnection(child);
        if (decider(rnd) <= addNodeMutation_Probability)
            MutateAddNode(child);
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