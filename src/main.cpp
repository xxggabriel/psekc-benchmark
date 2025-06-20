#include "include/DataManager.h"
#include "include/CPUProcessor.h"
#include "include/OMPProcessor.h"

#ifdef WITH_CUDA
#include "include/GPUProcessor.h"
#include <cuda_runtime.h>
#endif

#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <iomanip>
#include <omp.h>

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
bool check_consistency(const std::string& name1, const std::vector<double>& vec1,
                       const std::string& name2, const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        std::cout << "Inconsistência de tamanho: " << name1 << " (" << vec1.size() << ") vs "
                  << name2 << " (" << vec2.size() << ")" << std::endl;
        return false;
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(vec1[i] - vec2[i]));
    }

    bool is_consistent = max_diff < 1e-9;

    std::cout << "Verificando consistência: " << name1 << " vs " << name2 << std::endl;
    std::cout << "  -> Diferença máxima encontrada: " << std::scientific << max_diff << std::fixed << std::endl;
    std::cout << "  -> Resultados são consistentes? " << (is_consistent ? "Sim" : "Não") << std::endl;

    return is_consistent;
}

int main() {
    try {
        // --- 1. Configuração e Carregamento de Dados ---
        DataManager data_manager("../data/dirna_properties.csv", "../data/allData_8M.fasta");

        auto properties = data_manager.load_properties();
        auto sequence = data_manager.load_sequence();

        PseKNCParams params = {2, 10, 0.1};

        // --- 2. Definição dos Tamanhos para o Benchmark ---
        std::vector<long long> benchmark_sizes = {100, 1000, 10000, 100000};
        // Adiciona o tamanho completo da sequência como o teste final
        if(sequence.length() > 100000) {
            benchmark_sizes.push_back(sequence.length());
        }

        // --- 3. Execução do Benchmark ---
        BenchmarkRunner runner(sequence, params, properties);
        runner.run(benchmark_sizes);

    } catch (const std::exception& e) {
        std::cerr << "Ocorreu um erro fatal: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

/**
 * @brief Ponto de entrada principal do programa de benchmark.
 */
// int main() {
//     try {
//         DataManager data_manager("../data/dirna_properties.csv", "../data/allData_8M.fasta");
//         auto properties = data_manager.load_properties();
//         auto sequence = data_manager.load_sequence();
//
//         PseKNCParams params = {2, 2, 0.1};
//
//         std::cout << "\nIniciando benchmark..." << std::endl;
//         std::cout << "Tamanho da sequência: " << sequence.length() << " bases." << std::endl;
//         std::cout << "Parâmetros: K=" << params.k_value << ", LambdaMax=" << params.lambda_max << ", Weight=" << params.weight << std::endl;
//
//         std::cout << std::fixed << std::setprecision(4);
//
//
//
//         // --- 3. Benchmark da CPU Paralela (OpenMP) ---
//
//         // Obtém o número máximo de threads que o OpenMP usará
//         omp_set_num_threads(4);
//         int num_threads = omp_get_max_threads();
//
//         OMPProcessor omp_proc(properties, params);
//
//         std::cout << "\n--- Processando CPU Paralelo (OpenMP com " << num_threads << " threads) ---" << std::endl;
//         auto start_omp = std::chrono::high_resolution_clock::now();
//         auto vector_omp = omp_proc.process(sequence);
//         auto end_omp = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> duration_omp = end_omp - start_omp;
//         std::cout << "Tempo de execução: " << duration_omp.count() << " segundos." << std::endl;
//
//         // --- 4. Benchmark da GPU (CUDA) ---
//         double duration_gpu_val = -1.0;
//         std::vector<double> vector_gpu;
//         #ifdef WITH_CUDA
//             GPUProcessor gpu_proc(properties, params);
//             auto start_gpu = std::chrono::high_resolution_clock::now();
//             vector_gpu = gpu_proc.process(sequence);
//             cudaDeviceSynchronize();
//             auto end_gpu = std::chrono::high_resolution_clock::now();
//             duration_gpu_val = std::chrono::duration<double>(end_gpu - start_gpu).count();
//             std::cout << "\n--- Processamento GPU (CUDA) Concluído ---" << std::endl;
//             std::cout << "Tempo de execução: " << duration_gpu_val << " segundos." << std::endl;
//         #else
//             std::cout << "\nAVISO: Compilado sem suporte a CUDA. Benchmark da GPU ignorado." << std::endl;
//         #endif
//
//         // --- 2. Benchmark da CPU Sequencial ---
//         CPUProcessor cpu_proc(properties, params);
//         auto start_cpu = std::chrono::high_resolution_clock::now();
//         auto vector_cpu = cpu_proc.process(sequence);
//         auto end_cpu = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
//         std::cout << "\n--- Processamento CPU Sequencial Concluído ---" << std::endl;
//         std::cout << "Tempo de execução: " << duration_cpu.count() << " segundos." << std::endl;
//
//         // --- 5. Tabela de Resultados ---
//         std::cout << "\n--- Resultados Finais do Benchmark ---" << std::endl;
//         std::cout << std::left << std::setprecision(2);
//         std::cout << std::setw(29) << "Plataforma" << "| " << std::setw(9) << "Tempo (s)" << "| " << std::setw(19) << "Speedup vs CPU Seq." << "| " << "Eficiência" << std::endl;
//         std::cout << "-----------------------------|-----------|---------------------|-----------" << std::endl;
//
//         // Linha da CPU Sequencial
//         std::cout << std::setw(29) << "CPU Sequencial (1 thread)" << "| " << std::setw(9) << duration_cpu.count() << "| " << std::setw(19) << "1.00x" << "| " << "100.0%" << std::endl;
//
//         // Linha da CPU Paralela (OpenMP)
//         double speedup_omp = duration_cpu.count() / duration_omp.count();
//         double efficiency_omp = (speedup_omp / num_threads) * 100.0;
//         std::cout << std::setw(29) << "CPU Paralelo (OpenMP)" << "| " << std::setw(9) << duration_omp.count() << "| " << std::setw(19) << std::to_string(speedup_omp).substr(0,4) + "x" << "| " << std::to_string(efficiency_omp).substr(0,4) + "%" << std::endl;
//
//         // Linha da GPU
//         if(duration_gpu_val > 0) {
//             double speedup_gpu = duration_cpu.count() / duration_gpu_val;
//             std::cout << std::setw(29) << "GPU (CUDA Kernel)" << "| " << std::setw(9) << duration_gpu_val << "| " << std::setw(19) << std::to_string(speedup_gpu).substr(0,4) + "x" << "| " << "N/A" << std::endl;
//         }
//         std::cout << "----------------------------------------------------------------------" << std::endl;
//
//         // --- 6. Verificações de Consistência ---
//         std::cout << "\n--- Verificações de Consistência ---" << std::endl;
//         check_consistency("CPU Seq", vector_cpu, "CPU OMP", vector_omp);
//         if(duration_gpu_val > 0) {
//             check_consistency("CPU Seq", vector_cpu, "GPU CUDA", vector_gpu);
//         }
//
//     } catch (const std::exception& e) {
//         std::cerr << "Ocorreu um erro fatal: " << e.what() << std::endl;
//         return 1;
//     }
//
//     return 0;
// }
