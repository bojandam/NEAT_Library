#ifndef NEAT_LIBRARY_GENES_H
#define NEAT_LIBRARY_GENES_H
#include"includes.hpp"
#include<stack>
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
        std::vector<Connection> connections;
        uint numberOfInputs;
        uint numberOfOutputs;

        Genome(uint numberOfInputs,
            uint numberOfOutputs,
            std::uniform_real_distribution<double> & biasDistribution,
            std::mt19937 & rnd,
            const std::vector<Connection> & connections = {});

        bool link_would_create_loop(const Link & newLink);
        std::vector<std::vector<uint>> generateAdjList();
        std::map<uint, uint> MapNodeIdToIndex();
        uint NodeIdToIndex(uint NodeId);
    };


    Genome::Genome(uint numberOfInputs, uint numberOfOutputs, std::uniform_real_distribution<double> & biasDistribution,
        std::mt19937 & rnd, const std::vector<Connection> & connections)
        : numberOfInputs(numberOfInputs), numberOfOutputs(numberOfOutputs)
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

    std::map<uint, uint> Genome::MapNodeIdToIndex()
    {
        std::map<uint, uint> reId;
        for (int i = 0; i < nodes.size(); i++)
            reId[nodes[i].id] = i;
        return reId;
    }

    uint Genome::NodeIdToIndex(uint NodeId)
    {
        for (uint i = 0; i < nodes.size(); i++)
            if (NodeId == nodes[i].id)return i;
        return -1;
    }

    std::vector<std::vector<uint>> Genome::generateAdjList()
    {
        std::vector<std::vector<uint>>rez(nodes.size(), std::vector <uint>());
        std::map<uint, uint> reId = MapNodeIdToIndex();
        for (const Connection & connection : connections)
            rez[reId[connection.link.nodeInId]].push_back(reId[connection.link.nodeOutId]);
        return rez;
    }
    bool Genome::link_would_create_loop(const Link & newLink)
    {
        uint From = NodeIdToIndex(newLink.nodeInId);
        uint To = NodeIdToIndex(newLink.nodeOutId);
        if (nodes[From].nodeType == Node::OUTPUT || nodes[To].nodeType == Node::INPUT)
            return true;
        std::vector<std::vector<uint>> adjList = generateAdjList();

        for (uint to : adjList[From])
            if (to == To)
                return true;
        adjList[From].push_back(To);

        std::vector<bool> visited(nodes.size(), false), inPath(nodes.size(), false);
        std::stack<uint> Stack;
        for (uint i = 0; i < numberOfInputs; i++)
            Stack.push(i);
        while (!Stack.empty())
        {
            uint curNode = Stack.top();
            Stack.pop();
            visited[curNode] = true;
            inPath[curNode] = true;

            for (uint neighbour : adjList[curNode]) {
                if (inPath[neighbour])
                    return true;
                if (!visited[neighbour]) {
                    visited[neighbour] = true;
                    Stack.push(neighbour);
                }
            }
            inPath[curNode] = false;
        }
        return false;
    }


}

#endif