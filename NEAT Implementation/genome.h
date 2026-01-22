#ifndef NEAT_LIBRARY_GENES_H
#define NEAT_LIBRARY_GENES_H
#include"includes.h"

namespace NEAT {
    struct Genome
    {
        struct Node {
            enum NodeType { INPUT, HIDDEN, OUTPUT };
            uint id;
            NodeType nodeType;
            double bias;
            Node(uint id = 0, double bias = 0.5, NodeType nodeType = HIDDEN) : id(id), nodeType(nodeType), bias(bias) {}
        };
        struct Link {
            uint nodeInId;
            uint nodeOutId;
            Link(uint nodeInId = 0, uint nodeOutId = 0) :nodeInId(nodeInId), nodeOutId(nodeOutId) {}
        };
        struct Connection {
            Link link;
            double weight;
            bool isEnabled;
            uint innovationNumber;
            Connection(Link link = {}, double weight = 0, bool isEnabled = true, uint innovationNumber = 0)
                :link(link), weight(weight), isEnabled(isEnabled), innovationNumber(innovationNumber) {
            }
        };
    public:
        std::vector<Node> nodes;
        std::map<int, std::vector<Connection>> connections_AdjList;//[i]:-> connections that start at nodes[i]
        uint numberOfInputs;
        uint numberOfOutputs;

        Genome(uint numberOfInputs,
            uint numberOfOutputs,
            std::uniform_real_distribution<double> & biasDistribution,
            std::random_device & rnd,
            const std::vector<Connection> & Connections = {});

        bool link_would_create_loop(Link newLink);
    };


    Genome::Genome(uint numberOfInputs, uint numberOfOutputs, std::uniform_real_distribution<double> & biasDistribution,
        std::random_device & rnd, const std::vector<Connection> & Connections = {})
        : numberOfInputs(numberOfInputs), numberOfOutputs(numberOfOutputs)
    {
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















}
#endif