#include "NEAT.hpp"

namespace bNEAT {

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

    template<typename T> void NEAT::CrossoverVector(const std::vector<T> & fitter, const std::vector<T> & lessFit,
        std::vector<T> & childVector, uint(*getId)(const T & v))
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

    void NEAT::MarkForDeath(std::vector<Species> & speciesTracker) {
        std::vector<Individual * > GenerationIndividuals;
        for (Species & spicie : speciesTracker) {
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

    void NEAT::RemoveExtinctSpecies(SpeciesTracker & speciesTracker)
    {
        speciesTracker.erase(
            std::remove_if(speciesTracker.begin(), speciesTracker.end(), [](const Species & x) {return x.members.empty(); }),
            speciesTracker.end());
    }

    void NEAT::RemoveStagnantSpecies(SpeciesTracker & speciesTracker)
    {
        auto it = std::stable_partition(speciesTracker.begin(), speciesTracker.end(),
            [](const Species & x) {return (x.lifetime - x.lifetime_When_MaxFitness_Changed) >= 20; }
        );
        if (it != speciesTracker.begin() && it != speciesTracker.begin() + 1)  //if no progres was had in 20 generations they would've all died leaving no specie alive
            speciesTracker.erase(it, speciesTracker.end());
    }

    void NEAT::Speciate(SpeciesTracker & speciesTracker, Individual && individual)
    {
        bool joinedSpecie = false;
        for (Species & specie : speciesTracker) {
            if (CalculateCompatibility(individual.genome, specie.representitive.genome) < compatibilityTreshold) // check if individual fits in specie
            {
                specie.members.push_back(std::move(individual)); // individuals get moved to spiciation tracker (/table/organiser) 
                joinedSpecie = true;
                break;
            }
        }
        if (!joinedSpecie)
            speciesTracker.push_back(Species(std::move(individual)));//make a new specie
    }


    std::vector<Individual> NEAT::RunAlgorithm(
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
        SpeciesTracker speciesTracker;
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
                double genomeFitness = FitnessFunction(Phenotype(genome, activationFunction));
                Speciate(speciesTracker, Individual(std::move(genome), genomeFitness));
                GenerationFitness.push_back(genomeFitness);

            }
            Generation.clear();// full of junk after std::move()

            //3. Adjusted Fitness
            for (Species & spicie : speciesTracker) {
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
            RemoveStagnantSpecies(speciesTracker);
            MarkForDeath(speciesTracker);
            std::vector<Genome> NextGeneration;
            for (Species & specie : speciesTracker) {
                std::sort(specie.members.begin(), specie.members.end(), std::greater<Individual>());

                if (specie.members.size() > 5) //"champion of species with population larger than 5 are copied to the next gen"
                    NextGeneration.push_back(specie.members[0].genome);

                specie.KillMarkedSpecies();
            }
            RemoveExtinctSpecies(speciesTracker);

            //5.Crossover & Mutation
            while (NextGeneration.size() < generationSize * percentOf_MutateOnly_Children)
            {
                const Individual & parent = choose_within(choose_within(speciesTracker).members);
                Genome child(parent.genome);
                Mutate(child);
                NextGeneration.push_back(std::move(child));
            }

            while (NextGeneration.size() < generationSize)  //to do: make this select spicies
            {
                Species & specie = choose_within(speciesTracker);
                Individual * parentA = &choose_within(specie.members);
                Individual * parentB = (decider(rnd) <= interSpicie_Crossover_Chance) ? &choose_within(choose_within(speciesTracker).members) : &choose_within(specie.members);
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
            for (Species & specie : speciesTracker)
                specie.PrepareForNextGeneration();
            itterationsFinished++;

        } while (!TermanateCondition(std::move(GenerationFitness), itterationsFinished));
        //7.Termanation
        std::vector<Individual> resultingIndividuals;
        for (Genome & genome : Generation) {
            double genomeFitness = FitnessFunction(Phenotype(genome, activationFunction));
            resultingIndividuals.push_back(Individual(std::move(genome), genomeFitness));
        }
        std::sort(resultingIndividuals.begin(), resultingIndividuals.end(), std::greater<Individual>());
        return resultingIndividuals;
    }

};