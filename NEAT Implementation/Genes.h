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
    struct DisjointSetUninion
    {
        std::vector<int> parent;
        std::vector<int> size;

    };

    class NEAT {
    public:
        struct Genome {
            struct Node
            {
                enum NodeType { SENSOR, HIDDEN, OUTPUT };

                int id;
                NodeType nodeType;
                double bias;
            };
            struct Link
            {
                int nodeInId;
                int nodeOutId;
            };
            struct Connection {
                Link link;
                double weight;
                bool isEnabled;
                int innovationNumber;
            };


            std::vector<Node> nodes;
            std::vector<Connection> connections;
            int numberOfInputs;
            int numberOfOutputs;

            bool link_would_create_loop(Link newLink);
        private:

        };


    public:

        std::uniform_real_distribution<double> weightDistribution;
        std::uniform_real_distribution<double> nodeDistribution;
        std::random_device rnd;

        int nodeIdCounter = 0;
        int innovationCounter = 0;
        std::set<Genome::Link> innovationTracker;


        Genome Mutate_ChangeConnectionWeight(Genome Parent) {
            std::uniform_int_distribution<int> connectionSelector(0, Parent.connections.size());
            Parent.connections[connectionSelector(rnd)].weight = weightDistribution(rnd);
            return Parent;
        }
        Genome Mutate_ChangeNodeBias(Genome Parent) {
            std::uniform_int_distribution<int> nodeSelector(0, Parent.nodes.size());
            Parent.nodes[nodeSelector(rnd)].bias = weightDistribution(rnd);
            return Parent;
        }
        Genome Mutate_AddNode(Genome Parent) {
            std::uniform_int_distribution<int> connectionSelector(0, Parent.connections.size());
            int connectonIndex = connectionSelector(rnd);
            while (!Parent.connections[connectonIndex].isEnabled)//is this a good idea
                connectonIndex = connectionSelector(rnd);

            Parent.connections[connectonIndex].isEnabled = false;
            int newNodeId = nodeIdCounter++; // to do: Will this be affected by innovation tracking per generation, and will inovation tracking be per generation

        }
        Genome Mutate_AddConnection(Genome Parent) {
            std::uniform_int_distribution<int> nodeSelector(0, Parent.nodes.size());
            Genome::Link newLink{ Parent.nodes[nodeSelector(rnd)].id, Parent.nodes[nodeSelector(rnd)].id };
            while (Parent.link_would_create_loop(newLink)) // to do: Is while ok, or should it just fail if it would create a loop
            {
                newLink.nodeInId = Parent.nodes[nodeSelector(rnd)].id;
                newLink.nodeOutId = Parent.nodes[nodeSelector(rnd)].id;
            }
        }

        NEAT() :weightDistribution(-1, 1), nodeDistribution(0, 1) {}

    };
}
#endif //NEAT_LIBRARY_GENES_H