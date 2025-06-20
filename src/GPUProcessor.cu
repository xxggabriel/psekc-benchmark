#include "include/GPUProcessor.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>


#define CUDA_CHECK(err) { \
    cudaError_t e = err; \
    if (e != cudaSuccess) { \
        printf("Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}


__global__ void theta_kernel(const int* k_tuple_indices, const double* prop_table,
                           long long num_pairs, int lambda_i, int num_prop,
                           double* out_interactions) {
    long long i = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i < num_pairs) {
        int idx1 = k_tuple_indices[i];
        int idx2 = k_tuple_indices[i + lambda_i];

        double sum_sq_diff = 0.0;
        for (int p = 0; p < num_prop; ++p) {
            double val1 = prop_table[static_cast<long long>(idx1) * num_prop + p];
            double val2 = prop_table[static_cast<long long>(idx2) * num_prop + p];
            double diff = val1 - val2;
            sum_sq_diff += diff * diff;
        }
        out_interactions[i] = sum_sq_diff / num_prop;
    }
}


GPUProcessor::GPUProcessor(const PropertiesMap& properties, PseKNCParams params)
    : PseKNCProcessor(properties, params) {
    std::cout << "Processador GPU (CUDA Kernel) inicializado." << std::endl;
    setup_gpu_data();
}

GPUProcessor::~GPUProcessor() {
    std::cout << "Liberando memória da GPU..." << std::endl;
    if (d_property_table) CUDA_CHECK(cudaFree(d_property_table));
    if (d_k_tuple_indices) CUDA_CHECK(cudaFree(d_k_tuple_indices));
}

void GPUProcessor::setup_gpu_data() {
    // Mapeia k-tuple (string) para um índice (int)
    int index = 0;
    for (const auto& kt : sorted_ktuples) {
        k_tuple_to_index[kt] = index++;
    }

    num_properties = properties_map.begin()->second.size();
    size_t table_size = sorted_ktuples.size() * num_properties;
    std::vector<double> h_property_table(table_size);

    for (const auto& kt : sorted_ktuples) {
        int row = k_tuple_to_index[kt];
        const auto& props = properties_map.at(kt);
        for (size_t col = 0; col < num_properties; ++col) {
            h_property_table[static_cast<size_t>(row) * num_properties + col] = props[col];
        }
    }

    CUDA_CHECK(cudaMalloc(&d_property_table, table_size * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_property_table, h_property_table.data(), table_size * sizeof(double), cudaMemcpyHostToDevice));
}


double GPUProcessor::calculate_theta(const int* d_k_tuple_indices_seq, size_t num_k_tuples, int lambda_i) {
    long long num_pairs = static_cast<long long>(num_k_tuples) - lambda_i;
    if (num_pairs <= 0) return 0.0;

    double* d_interactions;
    CUDA_CHECK(cudaMalloc(&d_interactions, num_pairs * sizeof(double)));
    
    int threads_per_block = 256;
    long long grid_size = (num_pairs + threads_per_block - 1) / threads_per_block;

    theta_kernel<<<grid_size, threads_per_block>>>(
        d_k_tuple_indices_seq, d_property_table, num_pairs, lambda_i, num_properties, d_interactions
    );
    CUDA_CHECK(cudaGetLastError());

    // Usa Thrust para a redução (soma) na GPU, que é altamente otimizado
    thrust::device_ptr<double> d_interactions_ptr(d_interactions);
    double total_sum = thrust::reduce(d_interactions_ptr, d_interactions_ptr + num_pairs);
    
    CUDA_CHECK(cudaFree(d_interactions));

    return total_sum / num_pairs;
}


std::vector<double> GPUProcessor::process(const std::string& sequence) {
    auto k_tuples = prepare_ktuples(sequence);
    size_t num_k_tuples = k_tuples.size();
    
    
    std::vector<int> h_k_tuple_indices(num_k_tuples);
    for(size_t i = 0; i < num_k_tuples; ++i) {
        h_k_tuple_indices[i] = k_tuple_to_index[k_tuples[i]];
    }

    
    if (num_k_tuples > max_k_tuples_allocated) {
        if(d_k_tuple_indices) CUDA_CHECK(cudaFree(d_k_tuple_indices));
        CUDA_CHECK(cudaMalloc(&d_k_tuple_indices, num_k_tuples * sizeof(int)));
        max_k_tuples_allocated = num_k_tuples;
    }
    CUDA_CHECK(cudaMemcpy(d_k_tuple_indices, h_k_tuple_indices.data(), num_k_tuples * sizeof(int), cudaMemcpyHostToDevice));

    
    std::map<std::string, int> counts;
    for(const auto& kt : k_tuples) {
        counts[kt]++;
    }

    double total_k_tuples = k_tuples.size();
    std::vector<double> freq_vector;
    freq_vector.reserve(sorted_ktuples.size());

    for(const auto& kt : sorted_ktuples) {
        freq_vector.push_back(counts.count(kt) ? counts[kt] / total_k_tuples : 0.0);
    }
    
    std::vector<double> correlation_factors;
    correlation_factors.reserve(params.lambda_max);

    for(int i = 1; i <= params.lambda_max; ++i) {
        correlation_factors.push_back(calculate_theta(d_k_tuple_indices, num_k_tuples, i));
    }
    
    double correlation_sum = std::accumulate(correlation_factors.begin(), correlation_factors.end(), 0.0);
    double denominator = 1.0 + params.weight * correlation_sum;

    std::vector<double> final_vector;
    final_vector.reserve(freq_vector.size() + correlation_factors.size());

    for(double val : freq_vector) {
        final_vector.push_back(val / denominator);
    }

    for(double val : correlation_factors) {
        final_vector.push_back((params.weight * val) / denominator);
    }

    return final_vector;
}