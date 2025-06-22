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
    PseKNCProcessor(const PropertiesMap &properties, PseKNCParams params);

    virtual ~PseKNCProcessor() = default;

    virtual std::vector<double> process(const std::string &sequence) = 0;

    virtual std::vector<std::vector<double> > process_batch(const std::vector<std::string> &sequences);

    std::vector<std::string> get_feature_names() const;

    void set_params(PseKNCParams new_params);

protected:
    virtual std::vector<std::string> prepare_ktuples(const std::string &sequence);

    const PropertiesMap &properties_map;
    PseKNCParams params;
    std::vector<std::string> sorted_ktuples;
};
#endif // PSEKNCPROCESSOR_H
