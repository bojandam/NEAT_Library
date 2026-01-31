#ifndef NEAT_LIBRARY_GENES_H
#define NEAT_LIBRARY_GENES_H
#include"phenotype.hpp"
#include<stack>
namespace bNEAT {
    struct Phenotype;
    struct Node {
        enum NodeType { INPUT, HIDDEN, OUTPUT };
        uint id;
        NodeType nodeType;
        double bias;
        Node(uint id = 0, double bias = 0.5, NodeType nodeType = HIDDEN) : id(id), nodeType(nodeType), bias(bias) {}
    };
    struct Genome
    {

        struct Link {
            uint nodeInId;
            uint nodeOutId;
            Link(uint nodeInId = 0, uint nodeOutId = 0) :nodeInId(nodeInId), nodeOutId(nodeOutId) {}
            bool operator<(const Link & other) const {
                return (nodeInId < other.nodeInId || (nodeInId == other.nodeInId && nodeOutId < other.nodeOutId));
            }
            bool operator==(const Link & other) const {
                return nodeInId == other.nodeInId && nodeOutId == other.nodeOutId;
            }
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
        std::vector<Connection> connections;
        uint numberOfInputs;
        uint numberOfOutputs;

        Genome() = default;
        Genome(uint numberOfInputs,
            uint numberOfOutputs,
            std::normal_distribution<double> & biasDistribution,
            std::mt19937 & rnd,
            const std::vector<Connection> & connections = {});
        Genome(const Genome &) = default;
        Genome(Genome &&) = default;
        Genome & operator=(const Genome &) = default;


        bool link_would_create_loop(const Link & newLink);
        std::vector<std::vector<Phenotype::Link>> generateAdjList() const;
        std::map<uint, uint> MapNodeIdToIndex() const;
        uint NodeIdToIndex(uint NodeId) const;
    };


}

#endif