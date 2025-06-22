#include "include/DataManager.h"
#include <iostream>
#include <chrono>
#include <vector>
#include "include/BenchmarkRunner.h"
#include "include/DatasetGenerator.h"

// void run_benchmark() {
//     try {
//         DataManager data_manager("data/dirna_properties.csv", "data/mini-data.txt");
//         if (!data_manager.setup_data_files()) {
//              throw std::runtime_error("Falha ao obter ficheiros de dados para o benchmark.");
//         }
//         auto properties = data_manager.load_and_augment_properties();
//         auto sequences = data_manager.load_sequences();

//         PseKNCParams params = {2, 10, 0.1};

//         std::vector<long long> benchmark_sizes = {100, 1000, 10000, 100000};
//         if(sequences.size() > 100000 && sequences.size() > benchmark_sizes.back()) {
//             benchmark_sizes.push_back(sequences.size());
//         }

//         BenchmarkRunner runner(sequences, params, properties);
//         runner.run(benchmark_sizes);

//     } catch (const std::exception& e) {
//         std::cerr << "Ocorreu um erro durante o benchmark: " << e.what() << std::endl;
//     }
// }

void generate_datasets() {
    try {
        DatasetGenerator generator;
        std::filesystem::create_directories("generated_data");

        
        generator.generate_properties_file(2, 10, AlphabetType::DNA, "generated_data/dna_properties_k2.csv");

        generator.generate_fasta_file(1000000, 25, 35, AlphabetType::RNA, "generated_data/rna_sequences.fasta");

    } catch (const std::exception& e) {
        std::cerr << "Ocorreu um erro ao gerar datasets: " << e.what() << std::endl;
    }
}

void print_usage() {
    std::cout << "Uso: ./benchmark [modo]\n"
              << "Modos disponíveis:\n"
              << "  --scalability        (Padrão) Executa o benchmark de escalabilidade (threads e tamanho).\n"
              << "  --k-variation        Executa o benchmark variando o parâmetro K.\n"
              << "  --lambda-variation   Executa o benchmark variando o parâmetro Lambda.\n"
              << "  --generate-data      Gera datasets sintéticos de teste.\n";
}


int main(int argc, char* argv[]) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--generate") {
            std::cout << "Modo: Gerar Datasets" << std::endl;
            generate_datasets();
        } 

        DataManager data_manager("data/dirna_properties.csv", "data/mini-data.txt");
        if (!data_manager.setup_data_files()) {
             throw std::runtime_error("Falha ao obter ficheiros de dados para o benchmark.");
        }
        auto properties = data_manager.load_and_augment_properties();
        auto sequences = data_manager.load_sequences();
        
        BenchmarkRunner runner(sequences, properties);

        if (mode == "--scalability") {
            std::cout << "Modo: Benchmark de Escalabilidade" << std::endl;
            PseKNCParams params = {2, 10, 0.1};
            std::vector<long long> benchmark_sizes = {100, 1000, 10000, 100000};
            if(sequences.size() > 100000) benchmark_sizes.push_back(sequences.size());
            runner.run_scalability_benchmark(benchmark_sizes, params);
        } else if (mode == "--k-variation") {
            std::cout << "Modo: Benchmark de Variação de K" << std::endl;
            PseKNCParams base_params = {-1, 5, 0.1}; // K será substituído
            std::vector<int> k_values = {2, 3, 4, 5, 6}; // Valores de K para testar
            runner.run_k_variation_benchmark(k_values, 10000, base_params);
        } else if (mode == "--lambda-variation") {
            std::cout << "Modo: Benchmark de Variação de Lambda" << std::endl;
            PseKNCParams base_params = {2, -1, 0.1}; // Lambda será substituído
            std::vector<int> lambda_values = {1, 2, 3, 4, 5, 6};
            std::vector<long long> num_sequences = {100, 1000, 10000, 100000};
            runner.run_lambda_variation_benchmark(lambda_values, num_sequences, base_params);
        } else {
            print_usage();
        }

    } catch (const std::exception& e) {
        std::cerr << "Ocorreu um erro fatal: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
