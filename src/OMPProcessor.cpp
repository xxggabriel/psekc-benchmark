#include "include/OMPProcessor.h"
#include <iostream>
#include <map>
#include <numeric>
#include <algorithm>
#include <omp.h>

OMPProcessor::OMPProcessor(const PropertiesMap &properties, PseKNCParams params)
    : PseKNCProcessor(properties, params) {
    std::cout << "Processador CPU Paralelo (OpenMP) inicializado." << std::endl;
}

std::vector<std::string> OMPProcessor::prepare_ktuples(const std::string &sequence) {
    std::cout << "Gerando e filtrando k-tuples (método paralelo OpenMP)..." << std::endl;

    std::vector<std::vector<std::string> > thread_local_vectors;
    size_t num_k_tuples_total = 0;

#pragma omp parallel
    {
        // Cada thread tem o seu próprio vector para evitar condições de corrida
        std::vector<std::string> local_k_tuples;

#pragma omp for nowait
        for (size_t i = 0; i <= sequence.length() - params.k_value; ++i) {
            std::string kt = sequence.substr(i, params.k_value);
            if (std::binary_search(sorted_ktuples.begin(), sorted_ktuples.end(), kt)) {
                local_k_tuples.push_back(kt);
            }
        }

        // Usa uma secção crítica para juntar os resultados de forma segura
#pragma omp critical
        {
            num_k_tuples_total += local_k_tuples.size();
            thread_local_vectors.push_back(std::move(local_k_tuples));
        }
    }

    // Une os resultados de todas as threads num único vector
    std::vector<std::string> k_tuples_filtered;
    k_tuples_filtered.reserve(num_k_tuples_total);
    for (const auto &local_vec: thread_local_vectors) {
        k_tuples_filtered.insert(k_tuples_filtered.end(), local_vec.begin(), local_vec.end());
    }

    std::cout << "  -> Total de k-tuples válidos: " << k_tuples_filtered.size() << std::endl;
    return k_tuples_filtered;
}

double OMPProcessor::calculate_theta(const std::vector<std::string> &k_tuples, int lambda_i) {
    long long num_pairs = static_cast<long long>(k_tuples.size()) - lambda_i;
    if (num_pairs <= 0) return 0.0;

    double final_correlation_sum = 0.0;
    size_t num_properties = properties_map.begin()->second.size();

    // Vetor para armazenar as somas parciais de cada thread
    std::vector<double> partial_sums;

#pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        // Apenas uma thread inicializa o vetor de somas parciais
#pragma omp single
        {
            partial_sums.assign(num_threads, 0.0);
        }

        int thread_id = omp_get_thread_num();
        double local_sum = 0.0;


#pragma omp for
        for (long long i = 0; i < num_pairs; ++i) {
            const auto &vec1 = properties_map.at(k_tuples[i]);
            const auto &vec2 = properties_map.at(k_tuples[i + lambda_i]);
            double sum_sq_diff = 0.0;
            for (size_t p = 0; p < num_properties; ++p) {
                double diff = vec1[p] - vec2[p];
                sum_sq_diff += diff * diff;
            }
            local_sum += sum_sq_diff / num_properties;
        }
        // Cada thread guarda a sua soma parcial na sua posição do vetor
        partial_sums[thread_id] = local_sum;
    }

    // No final, a thread principal (sequencialmente) soma as somas parciais
    // Isto garante que a ordem da soma final seja sempre a mesma.
    for (double partial_sum: partial_sums) {
        final_correlation_sum += partial_sum;
    }

    return final_correlation_sum / num_pairs;
}

std::vector<double> OMPProcessor::process(const std::string &sequence) {
    auto k_tuples = prepare_ktuples(sequence); // Chama a nova versão paralela
    if (k_tuples.empty()) return {};

    std::map<std::string, int> counts;
    for (const auto &kt: k_tuples) counts[kt]++;
    double total_k_tuples = k_tuples.size();
    std::vector<double> freq_vector;
    for (const auto &kt: sorted_ktuples) freq_vector.push_back(counts.count(kt) ? counts[kt] / total_k_tuples : 0.0);
    std::vector<double> correlation_factors;
    for (int i = 1; i <= params.lambda_max; ++i) correlation_factors.push_back(calculate_theta(k_tuples, i));
    double correlation_sum = std::accumulate(correlation_factors.begin(), correlation_factors.end(), 0.0);
    double denominator = 1.0 + params.weight * correlation_sum;
    std::vector<double> final_vector;
    for (double val: freq_vector) final_vector.push_back(val / denominator);
    for (double val: correlation_factors) final_vector.push_back((params.weight * val) / denominator);
    return final_vector;
}
