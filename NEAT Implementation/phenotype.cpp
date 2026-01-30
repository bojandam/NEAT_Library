#include "phenotype.hpp"
#include "genome.hpp"
namespace bNEAT {

    std::vector<uint> topologicalSort(const std::vector<std::vector<Phenotype::Link>> & AdjList, uint nodeSize, uint numberOfInputs) {
        std::vector<bool> visited(nodeSize, false), finished(nodeSize, false);
        std::stack<uint> active;
        std::vector<uint> result;
        for (int j = 0; j < numberOfInputs; j++) {
            if (visited[j])
                continue; //how tf
            active.push(j);
            while (!active.empty()) {
                uint current = active.top();
                if (!visited[current]) {
                    visited[current] = true;
                    for (const Phenotype::Link & neighbour : AdjList[current]) {
                        if (!visited[neighbour.indexTo]) {
                            active.push(neighbour.indexTo);
                        }
                        else if (!finished[neighbour.indexTo]) {
                            return {};
                        }
                    }
                }
                else {
                    finished[current] = true;
                    result.push_back(current);
                    active.pop();
                }
            }
        }
        std::reverse(result.begin(), result.end());
        return result;
    }


    Phenotype::Phenotype(const Genome & genotype, ActivationFunction activationFunction) :
        AdjList(genotype.generateAdjList()),
        nodes(genotype.nodes),
        numberOfInputs(genotype.numberOfInputs),
        numberOfOutputs(genotype.numberOfOutputs),
        nodeCalculationOrder(topologicalSort(AdjList, nodes.size(), numberOfInputs)),
        activationFunction(activationFunction)
    {
    }

    std::vector<double>  Phenotype::Predict(std::vector<double> input) const
    {
        std::vector<double> nodeValues(nodes.size(), 0);
        for (int i = 0; i < numberOfInputs; i++) {
            nodeValues[i] = input[i];
        }
        for (int i : nodeCalculationOrder) {
            if (nodes[i].nodeType != Node::INPUT)
                nodeValues[i] = activationFunction(nodeValues[i] + nodes[i].bias);
            for (Link link : AdjList[i])
                nodeValues[link.indexTo] += nodeValues[i] * link.weight;
        }
        return std::vector<double>(nodeValues.begin() + numberOfInputs, nodeValues.begin() + numberOfInputs + numberOfOutputs);

    }
}

