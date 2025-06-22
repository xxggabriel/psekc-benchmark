#ifndef GPUPROCESSOR_H
#define GPUPROCESSOR_H
#include "PseKNCProcessor.h"
#include <vector>

class GPUProcessor : public PseKNCProcessor {
public:
    GPUProcessor(const PropertiesMap &properties, PseKNCParams params);

    ~GPUProcessor();

    std::vector<std::vector<double> > process_batch(const std::vector<std::string> &sequences) override;

private:
    std::vector<double> process(const std::string &sequence) override;

    void setup_gpu_data();

    double *d_property_table = nullptr;
    std::map<std::string, int> k_tuple_to_index;
    size_t num_properties;
};
#endif // GPUPROCESSOR_H
