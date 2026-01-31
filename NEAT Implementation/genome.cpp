#include "genome.hpp"

namespace bNEAT {
    Genome::Genome(
        uint numberOfInputs,
        uint numberOfOutputs,
        std::normal_distribution<double> & biasDistribution,
        std::mt19937 & rnd,
        const std::vector<Connection> & connections
    ) : numberOfInputs(numberOfInputs), numberOfOutputs(numberOfOutputs)
    {
        for (uint i = 0; i < numberOfInputs; i++) {
            nodes.push_back(Node(i, biasDistribution(rnd), Node::INPUT));
        }
        for (uint i = numberOfInputs; i - numberOfInputs < numberOfOutputs; i++) {
            nodes.push_back(Node(i, biasDistribution(rnd), Node::OUTPUT));
        }
        int number_of_nodes = nodes.size();
        for (const Connection & connection : connections) {
            if (connection.link.nodeInId < number_of_nodes && connection.link.nodeOutId < number_of_nodes) {
                this->connections.push_back(connection);
            }
        }
    }

    std::map<uint, uint> Genome::MapNodeIdToIndex() const
    {
        std::map<uint, uint> reId;
        for (int i = 0; i < nodes.size(); i++)
            reId[nodes[i].id] = i;
        return reId;
    }

    uint Genome::NodeIdToIndex(uint NodeId) const
    {
        for (uint i = 0; i < nodes.size(); i++)
            if (NodeId == nodes[i].id)return i;
        return -1;
    }

    std::vector < std::vector<Phenotype::Link>> Genome::generateAdjList() const
    {
        std::vector<std::vector<Phenotype::Link>>rez(nodes.size(), std::vector <Phenotype::Link>());
        std::map<uint, uint> reId = MapNodeIdToIndex();
        for (const Connection & connection : connections)
            if (connection.isEnabled)
                rez[reId[connection.link.nodeInId]].push_back({ reId[connection.link.nodeOutId],connection.weight });
        return rez;
    }

    bool Genome::link_would_create_loop(const Link & newLink)
    {
        uint From = NodeIdToIndex(newLink.nodeInId);
        uint To = NodeIdToIndex(newLink.nodeOutId);
        std::vector<std::vector<bNEAT::Phenotype::Link>> adjList(generateAdjList());

        if (nodes[From].nodeType == Node::OUTPUT || nodes[To].nodeType == Node::INPUT)
            return true;
        adjList[From].push_back({ To,0 });
        if (topologicalSort(adjList, nodes.size(), numberOfInputs).empty())
            return true;
        return false;
    }
}