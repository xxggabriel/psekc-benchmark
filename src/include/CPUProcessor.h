#ifndef CPUPROCESSOR_H
#define CPUPROCESSOR_H

#include "PseKNCProcessor.h"

class CPUProcessor : public PseKNCProcessor {
public:
    CPUProcessor(const PropertiesMap& properties, PseKNCParams params);
    std::vector<double> process(const std::string& sequence) override;

private:
    double calculate_theta(const std::vector<std::string>& k_tuples, int lambda_i);
};

#endif // CPUPROCESSOR_H