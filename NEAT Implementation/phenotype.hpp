#ifndef NEAT_LIBRARY_PHENOTYPE_H
#define NEAT_LIBRARY_PHENOTYPE_H
#include"includes.hpp"

namespace bNEAT
{
    struct Genome;
    struct Node;

    struct Phenotype
    {
        struct Link
        {
            uint indexTo;
            double weight;
        };
        uint numberOfInputs;
        uint numberOfOutputs;
        std::vector<std::vector<Link>> AdjList;
        std::vector<Node> nodes;
        std::vector<uint> nodeCalculationOrder;
        ActivationFunction activationFunction;
        Phenotype(const Genome &, ActivationFunction);
        std::vector<double> Predict(std::vector<double> input) const;
    };
    std::vector<uint> topologicalSort(const std::vector<std::vector<Phenotype::Link>> & AdjList, uint nodeSize, uint numberOfInputs);
}

#endif