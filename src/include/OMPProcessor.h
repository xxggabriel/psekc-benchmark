#ifndef OMPPROCESSOR_H
#define OMPPROCESSOR_H

#include "PseKNCProcessor.h"

class OMPProcessor : public PseKNCProcessor {
public:
    OMPProcessor(const PropertiesMap& properties, PseKNCParams params);
    std::vector<double> process(const std::string& sequence) override;

private:
    double calculate_theta(const std::vector<std::string>& k_tuples, int lambda_i);
};

#endif // OMPPROCESSOR_H