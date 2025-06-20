#ifndef GPUPROCESSOR_H
#define GPUPROCESSOR_H

#include "PseKNCProcessor.h"

class GPUProcessor : public PseKNCProcessor {
public:
    GPUProcessor(const PropertiesMap& properties, PseKNCParams params);
    ~GPUProcessor(); // Destrutor para liberar memória da GPU
    std::vector<double> process(const std::string& sequence) override;

private:
    void setup_gpu_data();
    double calculate_theta(const int* d_k_tuple_indices, size_t num_k_tuples, int lambda_i);

    // Ponteiros para a memória da GPU
    double* d_property_table = nullptr;
    int* d_k_tuple_indices = nullptr;
    
    std::map<std::string, int> k_tuple_to_index;
    size_t num_properties;
    size_t max_k_tuples_allocated = 0;
};

#endif // GPUPROCESSOR_H