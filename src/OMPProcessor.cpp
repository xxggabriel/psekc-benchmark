#include "include/OMPProcessor.h"
#include <iostream>
#include <map>
#include <numeric>
#include <cmath>
#include <omp.h> 

OMPProcessor::OMPProcessor(const PropertiesMap& properties, PseKNCParams params)
    : PseKNCProcessor(properties, params) {
    std::cout << "Processador CPU Paralelo (OpenMP) inicializado." << std::endl;
}

double OMPProcessor::calculate_theta(const std::vector<std::string>& k_tuples, int lambda_i) {
    long long num_pairs = static_cast<long long>(k_tuples.size()) - lambda_i;
    if (num_pairs <= 0) return 0.0;

    double correlation_sum = 0.0;
    size_t num_properties = properties_map.begin()->second.size();

    #pragma omp parallel for reduction(+:correlation_sum)
    for (long long i = 0; i < num_pairs; ++i) {
        const auto& vec1 = properties_map.at(k_tuples[i]);
        const auto& vec2 = properties_map.at(k_tuples[i + lambda_i]);
        
        double sum_sq_diff = 0.0;
        for (size_t p = 0; p < num_properties; ++p) {
            double diff = vec1[p] - vec2[p];
            sum_sq_diff += diff * diff;
        }
        correlation_sum += sum_sq_diff / num_properties;
    }
    return correlation_sum / num_pairs;
}

std::vector<double> OMPProcessor::process(const std::string& sequence) {
    auto k_tuples = prepare_ktuples(sequence);
    
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
        // A mágica do paralelismo acontece dentro desta função.
        correlation_factors.push_back(calculate_theta(k_tuples, i));
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
