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
                Node(int id = 0, double bias = 0.5, NodeType nodeType = HIDDEN) : id(id), nodeType(nodeType), bias(bias) {}
            };
            struct Link
            {
                int nodeInId;
                int nodeOutId;
                Link(int nodeInId = 0, int nodeOutId = 0) :nodeInId(nodeInId), nodeOutId(nodeOutId) {}
            };
            struct Connection {
                Link link;
                double weight;
                bool isEnabled;
                int innovationNumber;
                Connection(Link link = {}, double weight = 0, bool isEnabled = true, int innovationNumber = 0) :link(link), weight(weight), isEnabled(isEnabled), innovationNumber(innovationNumber) {}
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


        std::map<Genome::Link, int> innovationTracker_AddedConnections;
        std::map<Genome::Link, int> innovationTracker_AddedNode;// [old_link]=nodeId

        Genome Mutate_ChangeConnectionWeight(Genome Parent) {
            std::uniform_int_distribution<int> connection_selector(0, Parent.connections.size());
            Parent.connections[connection_selector(rnd)].weight = weightDistribution(rnd);
            return Parent;
        }
        Genome Mutate_ChangeNodeBias(Genome Parent) {
            std::uniform_int_distribution<int> node_selector(0, Parent.nodes.size());
            Parent.nodes[node_selector(rnd)].bias = weightDistribution(rnd);
            return Parent;
        }

        Genome Mutate_AddNode(Genome Parent) {
            std::uniform_int_distribution<int> connection_selector(0, Parent.connections.size());

            int connecton_index = connection_selector(rnd);

            while (!Parent.connections[connecton_index].isEnabled)//is this a good idea-----This will fail at the start when the NN is empty
                connecton_index = connection_selector(rnd);

            Genome::Link old_link = Parent.connections[connecton_index].link;

            int new_node_id;
            int innov_from;
            int innov_to;

            if (innovationTracker_AddedNode.find(old_link) != innovationTracker_AddedNode.end()) {
                new_node_id = innovationTracker_AddedNode[old_link];
                innov_from = innovationTracker_AddedConnections[{old_link.nodeInId, new_node_id}];
                innov_to = innovationTracker_AddedConnections[{new_node_id, old_link.nodeOutId}];
            }
            else {
                new_node_id = innovationTracker_AddedNode[old_link] = nodeIdCounter++;
                innov_from = innovationTracker_AddedConnections[{old_link.nodeInId, new_node_id}] = innovationCounter++;
                innov_to = innovationTracker_AddedConnections[{new_node_id, old_link.nodeOutId}] = innovationCounter++;
            }
            Genome::Node new_Node(new_node_id, weightDistribution(rnd), Genome::Node::HIDDEN);
            Genome::Connection new_connection_From({ old_link.nodeInId, new_node_id }, 1.0, true, innov_from);
            Genome::Connection new_connection_To({ new_node_id, old_link.nodeOutId }, Parent.connections[connecton_index].weight, true, innov_to);

            Parent.connections[connecton_index].isEnabled = false;
            Parent.nodes.push_back(new_Node);
            Parent.connections.push_back(new_connection_From);
            Parent.connections.push_back(new_connection_To);

        }

        Genome Mutate_AddConnection(Genome Parent) {
            std::uniform_int_distribution<int> node_selector(0, Parent.nodes.size());
            Genome::Link newLink{ Parent.nodes[node_selector(rnd)].id, Parent.nodes[node_selector(rnd)].id };
            while (Parent.link_would_create_loop(newLink)) // to do: Is while ok, or should it just fail if it would create a loop
            {
                newLink.nodeInId = Parent.nodes[node_selector(rnd)].id;
                newLink.nodeOutId = Parent.nodes[node_selector(rnd)].id;
            }
        }

        NEAT() :weightDistribution(-1, 1), nodeDistribution(0, 1) {}

    };
}
#endif //NEAT_LIBRARY_GENES_H