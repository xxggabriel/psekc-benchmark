#include "include/DataManager.h"
#include "include/BenchmarkRunner.h"
#include "include/DatasetGenerator.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <filesystem>
#include <string>


/**
 * @brief Imprime as instruções de uso do programa.
 */
void print_usage() {
    std::cout << "Uso: ./benchmark [modo]\n"
            << "Modos disponíveis:\n"
            << "  --scalability        (Padrão) Executa o benchmark de escalabilidade (threads e nº de sequências).\n"
            << "  --k-variation        Executa o benchmark variando o parâmetro K.\n"
            << "  --lambda-variation   Executa o benchmark variando o parâmetro Lambda.\n"
            << "  --generate-data      Gera datasets sintéticos de teste.\n";
}

/**
 * @brief Orquestra e executa o modo de benchmark selecionado.
 * @param mode O modo de benchmark a ser executado.
 */
void run_benchmarks(const std::string &mode) {
    const std::string DATA_URL = "https://github.com/salman-khan-mrd/Sprak-Pi-DNN/raw/main/Feature%20Extraction/DataSet/dirna.data";
    const std::string DATA_FILENAME = "dirna.data";
    const std::string SEQUENCE_URL = "https://raw.githubusercontent.com/salman-khan-mrd/Sprak-Pi-DNN/main/DataSet/allData%208M.txt";
    const std::string SEQUENCE_FILENAME = "../data/generated_data/rna_sequences.fasta";
    DataManager data_manager(DATA_URL, "../data/generated_data/rna_properties_k2.csv", // Usa o CSV convertido
                           SEQUENCE_URL, SEQUENCE_FILENAME);

    // if (!data_manager.setup_data_files()) {
    //     throw std::runtime_error("Falha ao obter ficheiros de dados para o benchmark.");
    // }

    auto properties = data_manager.load_and_augment_properties();
    auto sequences = data_manager.load_sequences();

    if (mode == "--scalability") {
        std::cout << "Modo: Benchmark de Escalabilidade" << std::endl;
        PseKNCParams params = {2, 10, 0.1};
        BenchmarkRunner runner(sequences, params, properties);
        std::vector<long long> benchmark_sizes = {100, 1000, 10000, 100000};
        if (sequences.size() > 100000 && sequences.size() > benchmark_sizes.back()) {
            benchmark_sizes.push_back(sequences.size());
        }
        runner.run_scalability_benchmark(benchmark_sizes);
    } else if (mode == "--k-variation") {
        std::cout << "Modo: Benchmark de Variação de K" << std::endl;
        PseKNCParams base_params = {-1, 5, 0.1}; // K será substituído
        BenchmarkRunner runner(sequences, base_params, properties);
        std::vector<int> k_values = {2, 3, 4, 5};
        runner.run_k_variation_benchmark(k_values, 10000, base_params);
    } else if (mode == "--lambda-variation") {
        std::cout << "Modo: Benchmark de Variação de Lambda" << std::endl;
        PseKNCParams base_params = {2, -1, 0.1};
        BenchmarkRunner runner(sequences, base_params, properties);
        std::vector<int> lambda_values = {1, 2, 4, 6, 8, 10};
        std::vector<long long> num_sequences = {10000, 100000};
        runner.run_lambda_variation_benchmark(lambda_values, num_sequences, base_params);
    } else {
        print_usage();
    }
}

/**
 * @brief Executa o gerador de datasets.
 */
void generate_datasets() {
    try {
        DatasetGenerator generator;
        std::filesystem::create_directories("../data/generated_data");

        // Exemplo 1: Gerar propriedades para dinucleotídeos de DNA
        generator.generate_properties_file(2, 10, AlphabetType::RNA, "../data/generated_data/rna_properties_k2.csv");
        generator.generate_properties_file(3, 10, AlphabetType::RNA, "../data/generated_data/rna_properties_k3.csv");
        generator.generate_properties_file(4, 10, AlphabetType::RNA, "../data/generated_data/rna_properties_k4.csv");
        generator.generate_properties_file(5, 10, AlphabetType::RNA, "../data/generated_data/rna_properties_k5.csv");
        generator.generate_properties_file(6, 10, AlphabetType::RNA, "../data/generated_data/rna_properties_k6.csv");

        // Exemplo 2: Gerar um ficheiro FASTA com 1_000_000 sequências de RNA
        generator.generate_fasta_file(1000000, 25, 35, AlphabetType::RNA, "../data/generated_data/rna_sequences.fasta");
    } catch (const std::exception &e) {
        std::cerr << "Ocorreu um erro ao gerar datasets: " << e.what() << std::endl;
    }
}

/**
 * @brief Ponto de entrada principal do programa.
 * @param argc Número de argumentos da linha de comando.
 * @param argv Vetor de argumentos da linha de comando.
 * @return 0 em caso de sucesso, 1 em caso de erro.
 */
int main(int argc, char *argv[]) {
    std::string mode = "--scalability"; // Modo padrão
    if (argc > 1) {
        mode = std::string(argv[1]);
    }

    try {
        if (mode == "--generate-data") {
            std::cout << "Modo: Gerar Datasets" << std::endl;
            generate_datasets();
        } else {
            run_benchmarks(mode);
        }
    } catch (const std::exception &e) {
        std::cerr << "Ocorreu um erro fatal: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
