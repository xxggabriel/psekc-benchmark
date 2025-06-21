#include "include/BenchmarkRunner.h"
#include "include/CPUProcessor.h"
#include "include/OMPProcessor.h"
#ifdef WITH_CUDA
#include "include/GPUProcessor.h"
#include <cuda_runtime.h>
#endif
#include <chrono>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>

BenchmarkRunner::BenchmarkRunner(std::vector<std::string> sequences, PseKNCParams params,
                                 const PropertiesMap &properties)
    : main_sequences(std::move(sequences)), params(params), properties(properties) {
    cpu_processor = std::make_unique<CPUProcessor>(properties, params);
    omp_processor = std::make_unique<OMPProcessor>(properties, params);
#ifdef WITH_CUDA
    gpu_processor = std::make_unique<GPUProcessor>(properties, params);
#endif
}

// Função auxiliar para verificar a consistência de duas matrizes de resultados
bool check_matrix_consistency(const std::string &name1, const std::vector<std::vector<double> > &matrix1,
                              const std::string &name2, const std::vector<std::vector<double> > &matrix2) {
    if (matrix1.size() != matrix2.size()) {
        std::cout << "  -> Inconsistência de tamanho (número de sequências): "
                << name1 << " (" << matrix1.size() << ") vs "
                << name2 << " (" << matrix2.size() << ")" << std::endl;
        return false;
    }

    double max_total_diff = 0.0;
    for (size_t i = 0; i < matrix1.size(); ++i) {
        const auto &vec1 = matrix1[i];
        const auto &vec2 = matrix2[i];
        if (vec1.size() != vec2.size()) {
            std::cout << "  -> Inconsistência de tamanho no vetor " << i << std::endl;
            return false;
        }
        for (size_t j = 0; j < vec1.size(); ++j) {
            max_total_diff = std::max(max_total_diff, std::abs(vec1[j] - vec2[j]));
        }
    }

    bool is_consistent = max_total_diff < 1e-9;
    std::cout << "Verificando consistência: " << name1 << " vs " << name2 << std::endl;
    std::cout << "  -> Diferença máxima encontrada: " << std::scientific << max_total_diff << std::fixed << std::endl;
    std::cout << "  -> Resultados são consistentes? " << (is_consistent ? "Sim" : "Não") << std::endl;
    return is_consistent;
}


void BenchmarkRunner::run(const std::vector<long long> &num_sequences_to_run) {
    std::vector<BenchmarkResult> all_results;

    for (long long num_seqs: num_sequences_to_run) {
        if (num_seqs > main_sequences.size()) {
            std::cout << "\nAviso: Número de sequências solicitado (" << num_seqs << ") é maior que o disponível ("
                    << main_sequences.size() << "). A ignorar." << std::endl;
            continue;
        }

        auto sequence_subset = std::vector<std::string>(main_sequences.begin(), main_sequences.begin() + num_seqs);

        std::cout << "\n======================================================================" << std::endl;
        std::cout << "Iniciando benchmark para " << num_seqs << " sequências" << std::endl;
        std::cout << "======================================================================" << std::endl;

        // --- Benchmark da GPU (CUDA) ---
        double duration_gpu_val = -1.0;
        std::vector<std::vector<double> > results_gpu;
#ifdef WITH_CUDA
        results_gpu.reserve(num_seqs);
        auto start_gpu = std::chrono::high_resolution_clock::now();
        for (const auto &seq: sequence_subset) {
            results_gpu.push_back(gpu_processor->process(seq));
        }
        cudaDeviceSynchronize();
        auto end_gpu = std::chrono::high_resolution_clock::now();
        duration_gpu_val = std::chrono::duration<double>(end_gpu - start_gpu).count();
        std::cout << "-> GPU (CUDA) Concluído em: " << duration_gpu_val << "s" << std::endl;
#endif

        // --- Benchmark da CPU Paralela (OpenMP) ---
        int num_threads = omp_get_max_threads();
        std::vector<std::vector<double> > results_omp(num_seqs);
        auto start_omp = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
        for (long long i = 0; i < num_seqs; ++i) {
            results_omp[i] = omp_processor->process(sequence_subset[i]);
        }
        auto end_omp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_omp = end_omp - start_omp;
        std::cout << "-> CPU Paralelo (OpenMP) Concluído em: " << duration_omp.count() << "s" << std::endl;


        // --- Benchmark da CPU Sequencial ---
        std::vector<std::vector<double> > results_cpu;
        results_cpu.reserve(num_seqs);
        auto start_cpu = std::chrono::high_resolution_clock::now();
        for (const auto &seq: sequence_subset) {
            results_cpu.push_back(cpu_processor->process(seq));
        }
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
        std::cout << "-> CPU Sequencial Concluído em: " << duration_cpu.count() << "s" << std::endl;

        // --- Armazenar Resultados e Verificar Consistência ---
        double speedup_omp = duration_cpu.count() / duration_omp.count();
        double efficiency_omp = (speedup_omp / num_threads) * 100.0;

        all_results.push_back({"CPU Sequencial", num_seqs, 1, duration_cpu.count(), 1.0, 100.0});
        all_results.push_back({
            "CPU Paralelo (OMP)", num_seqs, num_threads, duration_omp.count(), speedup_omp, efficiency_omp
        });

        if (duration_gpu_val >= 0) {
            double speedup_gpu = duration_cpu.count() / duration_gpu_val;
            all_results.push_back({"GPU (CUDA)", num_seqs, -1, duration_gpu_val, speedup_gpu, -1.0});
            check_matrix_consistency("CPU Seq", results_cpu, "GPU CUDA", results_gpu);
        }
        check_matrix_consistency("CPU Seq", results_cpu, "CPU OMP", results_omp);
    }

    ReportGenerator::print_table(all_results);
    ReportGenerator::save_to_csv(all_results, "../data/benchmark_report.csv");
}
