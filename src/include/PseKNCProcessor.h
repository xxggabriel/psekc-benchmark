#ifndef PSEKNCPROCESSOR_H
#define PSEKNCPROCESSOR_H

#include <vector>
#include <string>
#include "DataManager.h"

struct PseKNCParams {
    int k_value;
    int lambda_max;
    double weight;
};

class PseKNCProcessor {
public:
    PseKNCProcessor(const PropertiesMap& properties, PseKNCParams params);
    virtual ~PseKNCProcessor() = default;
    virtual std::vector<double> process(const std::string& sequence) = 0;

protected:
    std::vector<std::string> prepare_ktuples(const std::string& sequence);

    const PropertiesMap& properties_map;
    PseKNCParams params;
    std::vector<std::string> sorted_ktuples;
};

#endif // PSEKNCPROCESSOR_H