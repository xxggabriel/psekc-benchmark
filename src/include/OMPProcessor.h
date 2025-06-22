#ifndef OMPPROCESSOR_H
#define OMPPROCESSOR_H
#include "PseKNCProcessor.h"

class OMPProcessor : public PseKNCProcessor {
public:
    OMPProcessor(const PropertiesMap &properties, PseKNCParams params);

    std::vector<double> process(const std::string &sequence) override;

private:
    std::vector<std::string> prepare_ktuples(const std::string &sequence) override;

    double calculate_theta(const std::vector<std::string> &k_tuples, int lambda_i);
};
#endif // OMPPROCESSOR_H
