#include "include/ReportGenerator.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>

void ReportGenerator::print_table(const std::vector<BenchmarkResult>& results, bool show_k_lambda) {
    std::cout << "\n--- Resultados Finais do Benchmark ---" << std::endl;
    std::cout << std::left << std::setprecision(4);

    if (show_k_lambda) {
        std::cout << std::setw(20) << "Plataforma" << "| " << std::setw(15) << "Tamanho Seq." << "| " << std::setw(3) << "K" << "| " << std::setw(4) << "LMax" << "| " << std::setw(10) << "Tempo (s)" << std::endl;
        std::cout << "---------------------|---------------|---|----|-----------" << std::endl;
        for (const auto& res : results) {
            std::cout << std::setw(20) << res.platform << "| " << std::setw(15) << res.sequence_size << "| " << std::setw(3) << res.k_value << "| " << std::setw(4) << res.lambda_max << "| " << std::setw(10) << res.duration_sec << std::endl;
        }
    } else {
        std::cout << std::setw(29) << "Plataforma" << "| " << std::setw(15) << "Tamanho Seq." << "| " << std::setw(9) << "Threads" << "| " << std::setw(11) << "Tempo (s)" << "| " << std::setw(19) << "Speedup vs CPU Seq." << "| " << "Eficiência (%)" << std::endl;
        std::cout << "-----------------------------|---------------|-----------|-------------|---------------------|---------------" << std::endl;
        for (const auto& res : results) {
            std::cout << std::setw(29) << res.platform << "| " << std::setw(15) << res.sequence_size << "| " << std::setw(9) << res.num_threads << "| " << std::setw(11) << res.duration_sec << "| " << std::setw(19) << (std::to_string(res.speedup).substr(0, 5) + "x") << "| ";
            if (res.efficiency_percent >= 0) std::cout << res.efficiency_percent; else std::cout << "N/A";
            std::cout << std::endl;
        }
    }
}

void ReportGenerator::save_to_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro: Não foi possível criar o ficheiro de relatório: " << filename << std::endl;
        return;
    }

        file << "Platform,SequenceSize,NumThreads,K_Value,Lambda_Max,Duration_sec,Speedup,Efficiency_percent\n";


    for (const auto& res : results) {
        file << res.platform << "," 
                << res.sequence_size 
                << "," 
                << res.num_threads 
                << "," 
                << res.k_value 
                << "," 
                << res.lambda_max 
                << "," 
                << res.duration_sec 
                << "," 
                << res.speedup 
                << "," 
                << (res.efficiency_percent >= 0 ? std::to_string(res.efficiency_percent) : "N/A") 
                << "\n";
    }
    
    std::cout << "\nRelatório de benchmark guardado em '" << filename << "'." << std::endl;
}

void ReportGenerator::save_feature_matrix(
    std::string base_dir,
    const std::vector<std::vector<double>>& matrix,
    const std::vector<std::string>& feature_names,
    const std::vector<std::string>& sequence_ids,
    const std::string& platform_name,
    long long num_sequences)
{
    if (matrix.empty()) return;
    if (matrix.size() != sequence_ids.size()) {
        std::cerr << "Erro: O número de matrizes de características não corresponde ao número de IDs de sequência." << std::endl;
        return;
    }
    if (feature_names.empty()) {
        std::cerr << "Erro: A lista de nomes de características está vazia." << std::endl;
        return;
    }

    std::filesystem::create_directories(base_dir);

    std::string platform_dir = base_dir + "/" + platform_name;
    std::filesystem::create_directories(platform_dir);

    std::string filename = platform_dir + "/features_" + std::to_string(num_sequences) + "_seqs.csv";

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro: Não foi possível criar o ficheiro de características: " << filename << std::endl;
        return;
    }

    // Escreve o cabeçalho, agora com "ID"
    file << "ID,";
    for (size_t i = 0; i < feature_names.size(); ++i) {
        file << feature_names[i] << (i == feature_names.size() - 1 ? "" : ",");
    }
    file << "\n";

    // Escreve cada linha da matriz, começando com o ID
    file << std::fixed << std::setprecision(10);
    for (size_t row_idx = 0; row_idx < matrix.size(); ++row_idx) {
        file << sequence_ids[row_idx] << ","; // Escreve o ID
        const auto& feature_vector = matrix[row_idx];
        for (size_t i = 0; i < feature_vector.size(); ++i) {
            file << feature_vector[i] << (i == feature_vector.size() - 1 ? "" : ",");
        }
        file << "\n";
    }

    std::cout << "-> Matriz de características salva em '" << filename << "'" << std::endl;
}
