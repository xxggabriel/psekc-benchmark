#include "include/ReportGenerator.h"

#include <fstream>
#include <iomanip>

void ReportGenerator::print_table(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n--- Resultados Finais do Benchmark ---" << std::endl;
    std::cout << std::left << std::setprecision(4);
    std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::setw(29) << "Plataforma" << "| "
              << std::setw(15) << "Tamanho da Seq." << "| "
              << std::setw(11) << "Tempo (s)" << "| "
              << std::setw(19) << "Speedup vs CPU Seq." << "| "
              << "Eficiência (%)" << std::endl;
    std::cout << "-----------------------------|---------------|-------------|---------------------|---------------" << std::endl;

    for (const auto& res : results) {
        std::cout << std::setw(29) << res.platform << "| "
                  << std::setw(15) << res.sequence_size << "| "
                  << std::setw(11) << res.duration_sec << "| "
                  << std::setw(19) << (std::to_string(res.speedup).substr(0, 5) + "x") << "| ";
        if (res.efficiency_percent >= 0) {
            std::cout << res.efficiency_percent;
        } else {
            std::cout << "N/A";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------------------------------------------------------------------------------------" << std::endl;
}

void ReportGenerator::save_to_csv(const std::vector<BenchmarkResult>& results, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro: Não foi possível criar o ficheiro de relatório: " << filename << std::endl;
        return;
    }

    // Escreve o cabeçalho
    file << "Platform,SequenceSize,NumThreads,Duration_sec,Speedup,Efficiency_percent\n";

    // Escreve os dados
    for (const auto& res : results) {
        file << res.platform << ","
             << res.sequence_size << ","
             << res.num_threads << ","
             << res.duration_sec << ","
             << res.speedup << ","
             << (res.efficiency_percent >= 0 ? std::to_string(res.efficiency_percent) : "N/A") << "\n";
    }

    std::cout << "\nRelatório de benchmark guardado com sucesso em '" << filename << "'." << std::endl;
}
