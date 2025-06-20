#include "include/PseKNCProcessor.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>

PseKNCProcessor::PseKNCProcessor(const PropertiesMap& properties, PseKNCParams params)
    : properties_map(properties), params(std::move(params)) {
    for (const auto& pair : properties_map) {
        sorted_ktuples.push_back(pair.first);
    }
    std::sort(sorted_ktuples.begin(), sorted_ktuples.end());
}

std::vector<std::string> PseKNCProcessor::prepare_ktuples(const std::string& sequence) {
    std::cout << "Gerando e filtrando k-tuples..." << std::endl;

    std::vector<std::string> k_tuples_raw;
    if (sequence.length() >= static_cast<size_t>(params.k_value)) {
        k_tuples_raw.reserve(sequence.length() - params.k_value + 1);
        for (size_t i = 0; i <= sequence.length() - params.k_value; ++i) {
            k_tuples_raw.push_back(sequence.substr(i, params.k_value));
        }
    }

    std::cout << "Total de k-tuples gerados: " << k_tuples_raw.size() << std::endl;

    std::vector<std::string> k_tuples_filtered;
    k_tuples_filtered.reserve(k_tuples_raw.size());

    for (const auto& kt : k_tuples_raw) {
        if (std::binary_search(sorted_ktuples.begin(), sorted_ktuples.end(), kt)) {
            k_tuples_filtered.push_back(kt);
        }
    }

    std::cout << "Total de k-tuples válidos: " << k_tuples_filtered.size() << std::endl;
    if (k_tuples_filtered.empty() && !sequence.empty()) {
        throw std::runtime_error("Nenhum k-tuple válido foi encontrado na sequência.");
    }
    return k_tuples_filtered;
}

