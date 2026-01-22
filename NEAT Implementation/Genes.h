#ifndef NEAT_LIBRARY_GENES_H
#define NEAT_LIBRARY_GENES_H
#include<vector>
#include<set>
#include<map>
#include<cmath>
#include<random> 
namespace NEAT {
    namespace StandardActivationFunctions {
        double Sigmoid(double x) {
            return 1 / (1 + std::exp(-x));
        }
    }
    typedef double (*ActivationFunction) (double);
    typedef unsigned int uint;
    class NEAT {
    public:
        struct Genome {
            struct Node
            {
                enum NodeType { INPUT, HIDDEN, OUTPUT };
                uint id;
                NodeType nodeType;
                double bias;
                Node(uint id = 0, double bias = 0.5, NodeType nodeType = HIDDEN) : id(id), nodeType(nodeType), bias(bias) {}
            };
            struct Link
            {
                uint nodeInId;
                uint nodeOutId;
                Link(uint nodeInId = 0, uint nodeOutId = 0) :nodeInId(nodeInId), nodeOutId(nodeOutId) {}
            };
            struct Connection {
                Link link;
                double weight;
                bool isEnabled;
                uint innovationNumber;
                Connection(Link link = {}, double weight = 0, bool isEnabled = true, uint innovationNumber = 0) :link(link), weight(weight), isEnabled(isEnabled), innovationNumber(innovationNumber) {}
            };


            std::vector<Node> nodes;
            std::map<int, std::vector<Connection>> connections_AdjList;
            uint numberOfInputs;
            uint numberOfOutputs;

            Genome(uint numberOfInputs, uint numberOfOutputs, std::uniform_real_distribution<double> & biasDistribution, std::random_device & rnd, const std::vector<Connection> & Connections = {}) :
                numberOfInputs(numberOfInputs), numberOfOutputs(numberOfOutputs) {
                for (uint i = 0; i < numberOfInputs; i++) {
                    nodes.push_back(Node(i, biasDistribution(rnd), Node::INPUT));
                }
                for (uint i = numberOfInputs; i - numberOfInputs < numberOfOutputs; i++) {
                    nodes.push_back(Node(i, biasDistribution(rnd), Node::OUTPUT));
                }
                int number_of_nodes = nodes.size();
                for (const Connection & connection : Connections) {
                    if (connection.link.nodeInId < number_of_nodes && connection.link.nodeOutId < number_of_nodes) {
                        this->connections_AdjList[connection.link.nodeInId].push_back(connection);
                    }
                }
            }

            bool link_would_create_loop(Link newLink);
        private:

        };
    protected:
        std::uniform_real_distribution<double> weightDistribution;
        std::uniform_real_distribution<double> nodeDistribution;
        std::random_device rnd;

        uint nodeIdCounter;
        uint innovationCounter = 0;

        std::map<Genome::Link, int> innovationTracker_AddedConnections;
        std::map<Genome::Link, int> innovationTracker_AddedNode;// [old_link]=nodeId

        uint numOfInputsInNN;
        uint numOfOutputsInNN;
        uint generationSize;

        std::allocator<Genome> StartingPopulationAllocator;
    public:
        NEAT(uint numOfInputsInNN, uint numOfOutputsInNN, uint generationSize = 1000) :weightDistribution(-1, 1), nodeDistribution(0, 1),
            numOfInputsInNN(numOfInputsInNN), numOfOutputsInNN(numOfOutputsInNN), generationSize(generationSize) {
            nodeIdCounter = numOfInputsInNN + numOfOutputsInNN;
        }

        void StartAlgorithm(const std::vector<Genome::Link> & StarterLinks = {}) {

            //Initalization
            std::vector<Genome::Connection> StarterConnections;
            for (Genome::Link link : StarterLinks) {
                StarterConnections.push_back(Genome::Connection(link, weightDistribution(rnd), true, innovationCounter++));
            }
            std::vector<Genome> Generation(generationSize, Genome(numOfInputsInNN, numOfOutputsInNN, weightDistribution, rnd, StarterConnections));

            //to do:Termanate condition --- how will I implement this
                //Calculate Fitness

                //Selection

                //Crossover

                //Mutation


        }



    };

    bool NEAT::Genome::link_would_create_loop(Link newLink)
    {
        return false;
    }
}
#endif //NEAT_LIBRARY_GENES_H