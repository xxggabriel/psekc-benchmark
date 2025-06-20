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

/**
 * @brief Compara dois vetores de double e verifica se são consistentes
 * dentro de uma tolerância definida.
 * @param name1 Nome do primeiro vetor para exibição.
 * @param vec1 O primeiro vetor.
 * @param name2 Nome do segundo vetor para exibição.
 * @param vec2 O segundo vetor.
 * @return true se os vetores forem consistentes, false caso contrário.
 */
bool check_consistency(const std::string &name1, const std::vector<double> &vec1,
                       const std::string &name2, const std::vector<double> &vec2) {
    if (vec1.size() != vec2.size()) {
        std::cout << "Inconsistência de tamanho: " << name1 << " (" << vec1.size() << ") vs "
                << name2 << " (" << vec2.size() << ")" << std::endl;
        return false;
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(vec1[i] - vec2[i]));
    }

    // Uma tolerância realista para comparações de ponto flutuante de 64 bits (double)
    // entre diferentes arquiteturas e ordens de operação.
    bool is_consistent = max_diff < 1e-9;

    std::cout << "Verificando consistência: " << name1 << " vs " << name2 << std::endl;
    std::cout << "  -> Diferença máxima encontrada: " << std::scientific << max_diff << std::fixed << std::endl;
    std::cout << "  -> Resultados são consistentes? " << (is_consistent ? "Sim" : "Não") << std::endl;

    return is_consistent;
}

/**
 * @brief Ponto de entrada principal do programa de benchmark.
 */
int main() {
    try {
        DataManager data_manager("../data/dirna_properties.csv", "../data/allData_8M.fasta");
        auto properties = data_manager.load_properties();
        auto sequence = data_manager.load_sequence();

        PseKNCParams params = {2, 10, 0.1};

        std::cout << "\nIniciando benchmark..." << std::endl;
        std::cout << "Tamanho da sequência: " << sequence.length() << " bases." << std::endl;
        std::cout << "Parâmetros: K=" << params.k_value << ", LambdaMax=" << params.lambda_max << ", Weight=" << params.
                weight << std::endl;

        std::cout << std::fixed << std::setprecision(4); // Formatação para tempos

        // --- Benchmark da GPU ---
        double duration_gpu_val = -1.0;
        std::vector<double> vector_gpu;
#ifdef WITH_CUDA
        GPUProcessor gpu_proc(properties, params);
        auto start_gpu = std::chrono::high_resolution_clock::now();
        vector_gpu = gpu_proc.process(sequence);
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        duration_gpu_val = std::chrono::duration<double>(end_gpu - start_gpu).count();
        std::cout << "\n--- Processamento GPU (CUDA) Concluído ---" << std::endl;
        std::cout << "Tempo de execução: " << duration_gpu_val << " segundos." << std::endl;
#else
        std::cout << "\nCompilado sem suporte a CUDA. Benchmark da GPU ignorado." << std::endl;
#endif

        // --- Benchmark da CPU Paralela (OpenMP) ---
        OMPProcessor omp_proc(properties, params);
        auto start_omp = std::chrono::high_resolution_clock::now();
        auto vector_omp = omp_proc.process(sequence);
        auto end_omp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_omp = end_omp - start_omp;
        std::cout << "\n--- Processamento CPU Paralelo (OpenMP) Concluído ---" << std::endl;
        std::cout << "Tempo de execução: " << duration_omp.count() << " segundos." << std::endl;

        // --- Benchmark da CPU Sequencial ---
        CPUProcessor cpu_proc(properties, params);
        auto start_cpu = std::chrono::high_resolution_clock::now();
        auto vector_cpu = cpu_proc.process(sequence);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
        std::cout << "\n--- Processamento CPU Sequencial Concluído ---" << std::endl;
        std::cout << "Tempo de execução: " << duration_cpu.count() << " segundos." << std::endl;

        // --- Tabela de Resultados ---
        std::cout << "\n--- Resultados Finais do Benchmark ---" << std::endl;
        std::cout << std::setprecision(2);
        std::cout << "Plataforma                | Tempo (s) | Speedup vs CPU Seq." << std::endl;
        std::cout << "--------------------------|-----------|---------------------" << std::endl;
        std::cout << "CPU Sequencial (1 thread) | " << std::setw(9) << duration_cpu.count() << " | 1.00x" << std::endl;
        std::cout << "CPU Paralelo (OpenMP)     | " << std::setw(9) << duration_omp.count() << " | " << (
            duration_cpu.count() / duration_omp.count()) << "x" << std::endl;
        if (duration_gpu_val > 0) {
            std::cout << "GPU (CUDA Kernel)         | " << std::setw(9) << duration_gpu_val << " | " << (
                duration_cpu.count() / duration_gpu_val) << "x" << std::endl;
        }
        std::cout << "------------------------------------------------------------" << std::endl;

        // --- Verificações de Consistência ---
        std::cout << "\n--- Verificações de Consistência ---" << std::endl;
        check_consistency("CPU Seq", vector_cpu, "CPU OMP", vector_omp);
        if (duration_gpu_val > 0) {
            check_consistency("CPU Seq", vector_cpu, "GPU CUDA", vector_gpu);
        }
    } catch (const std::exception &e) {
        std::cerr << "Ocorreu um erro fatal: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
