#include "include/PseKNCProcessor.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>

PseKNCProcessor::PseKNCProcessor(const PropertiesMap &properties, PseKNCParams params)
    : properties_map(properties), params(std::move(params)) {
    for (const auto &pair: properties_map) {
        sorted_ktuples.push_back(pair.first);
    }
    std::sort(sorted_ktuples.begin(), sorted_ktuples.end());
}

void PseKNCProcessor::set_params(PseKNCParams new_params) {
    this->params = new_params;
}


std::vector<std::vector<double>> PseKNCProcessor::process_batch(const std::vector<std::string>& sequences) {
    std::vector<std::vector<double>> results;
    results.reserve(sequences.size());
    for (const auto& seq : sequences) {
        results.push_back(process(seq));
    }
    return results;
}

std::vector<std::string> PseKNCProcessor::get_feature_names() const {
    std::vector<std::string> names = sorted_ktuples;
    for (int i = 1; i <= params.lambda_max; ++i) {
        names.push_back("theta_" + std::to_string(i));
    }
    return names;
}

std::vector<std::string> PseKNCProcessor::prepare_ktuples(const std::string &sequence) {
    std::vector<std::string> k_tuples_filtered;

    if (sequence.length() >= static_cast<size_t>(params.k_value)) {
        k_tuples_filtered.reserve(sequence.length() - params.k_value + 1);

        for (size_t i = 0; i <= sequence.length() - params.k_value; ++i) {
            std::string kt = sequence.substr(i, params.k_value);

            if (std::binary_search(sorted_ktuples.begin(), sorted_ktuples.end(), kt)) {
                k_tuples_filtered.push_back(kt);
            }
        }
    }

    return k_tuples_filtered;
}