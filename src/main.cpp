#include "include/DataManager.h"
#include <iostream>
#include <chrono>
#include <vector>
#include "include/BenchmarkRunner.h"

/**
 * @brief Compara dois vetores de double e verifica se são consistentes
 * dentro de uma tolerância definida.
 * @param name1 Nome do primeiro vetor para exibição.
 * @param vec1 O primeiro vetor.
 * @param name2 Nome do segundo vetor para exibição.
 * @param vec2 O segundo vetor.
 * @return true se os vetores forem consistentes, false caso contrário.
 */


int main() {
    try {
        DataManager data_manager("../data/dirna_properties.csv", "../data/allData_8M.fasta");

        auto properties = data_manager.load_and_augment_properties();
        auto sequences = data_manager.load_sequences();

        PseKNCParams params = {2, 10, 0.1};

        std::vector<long long> benchmark_sizes = {100, 1000, 10000};
        if(sequences.size() > 10000) {
            benchmark_sizes.push_back(sequences.size());
        }

        BenchmarkRunner runner(sequences, params, properties);
        runner.run(benchmark_sizes);

    } catch (const std::exception& e) {
        std::cerr << "Ocorreu um erro fatal: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}