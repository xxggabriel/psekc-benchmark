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

#include "include/ReportGenerator.h"

BenchmarkRunner::BenchmarkRunner(std::vector<SequenceData> sequences,
                                 PseKNCParams params,
                                 const PropertiesMap &properties)
    : main_sequences(std::move(sequences)),
      params(params),
      properties(properties) {
    cpu_processor = std::make_unique<CPUProcessor>(properties, params);
    omp_processor = std::make_unique<OMPProcessor>(properties, params);
#ifdef WITH_CUDA
    gpu_processor = std::make_unique<GPUProcessor>(properties, params);
#endif
}

bool check_matrix_consistency(const std::string &name1, const std::vector<std::vector<double> > &matrix1,
                              const std::string &name2, const std::vector<std::vector<double> > &matrix2) {
    if (matrix1.empty() || matrix2.empty()) {
        std::cout << "Aviso: Impossível verificar consistência pois um dos resultados está vazio." << std::endl;
        return false;
    }
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
    const auto feature_names = cpu_processor->get_feature_names();

    for (long long num_seqs: num_sequences_to_run) {
        if (num_seqs > (long long) main_sequences.size()) {
            std::cout << "\nAviso: Solicitação de " << num_seqs
                    << " sequências, mas só há " << main_sequences.size()
                    << ". Ignorando.\n";
            continue;
        }

        auto sequence_pairs_subset = std::vector<SequenceData>(main_sequences.begin(),
                                                               main_sequences.begin() + num_seqs);
        std::vector<std::string> sequence_subset;
        std::vector<std::string> id_subset;
        sequence_subset.reserve(num_seqs);
        id_subset.reserve(num_seqs);
        for (const auto &pair: sequence_pairs_subset) {
            id_subset.push_back(pair.first);
            sequence_subset.push_back(pair.second);
        }

        bool cpu_sequential_benchmark = true;
        bool omp_benchmark = true;
        bool gpu_benchmark = true;

        std::cout << "\n===== Benchmark: " << num_seqs << " sequências =====" << std::endl;

        double time_cpu = -1.0;
        double time_gpu = -1.0;

        std::vector<std::vector<double> > results_cpu;
        std::vector<std::vector<double> > results_omp;
        std::vector<std::vector<double> > results_gpu;

        // --- Benchmark da CPU Sequencial ---
        if (cpu_sequential_benchmark) {
            results_cpu.resize(num_seqs);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (long long i = 0; i < num_seqs; ++i) {
                results_cpu[i] = cpu_processor->process(sequence_subset[i]);
            }
            auto t1 = std::chrono::high_resolution_clock::now();

            time_cpu = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "-> CPU Seq: " << time_cpu << " s" << std::endl;
            all_results.push_back({"CPU Seq", num_seqs, 1, time_cpu, 1.0, 100.0});
            ReportGenerator::save_feature_matrix(
                "../data/cpu_results",
                results_cpu,
                feature_names,
                id_subset,
                "CPU_Sequencial",
                num_seqs
            );
        }

        // --- Benchmark da CPU Paralela (OpenMP) ---
        if (omp_benchmark) {
            int max_threads = omp_get_max_threads();
            for (int nt = 2; nt <= max_threads; nt *= 2) {
                omp_set_num_threads(nt);
                results_omp.assign(num_seqs, {}); // Limpa e redimensiona
                auto t2 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
                for (long long i = 0; i < num_seqs; ++i) {
                    results_omp[i] = omp_processor->process(sequence_subset[i]);
                }
                auto t3 = std::chrono::high_resolution_clock::now();
                double time_omp = std::chrono::duration<double>(t3 - t2).count();

                double speedup = (time_cpu > 0) ? (time_cpu / time_omp) : 0.0;
                double efficiency = (time_cpu > 0) ? ((speedup / nt) * 100.0) : 0.0;

                std::cout << "-> OMP (" << nt << " threads): " << time_omp
                        << " s | Speedup=" << speedup
                        << " | Eff=" << efficiency << "%" << std::endl;
                all_results.push_back({"OpenMP", num_seqs, nt, time_omp, speedup, efficiency});
                ReportGenerator::save_feature_matrix(
                    "../data/omp_results",
                    results_omp,
                    feature_names,
                    id_subset,
                    "OMP_" + std::to_string(nt) + "_threads",
                    num_seqs
                );
            }

            if (cpu_sequential_benchmark) {
                check_matrix_consistency("CPU Seq", results_cpu, "OpenMP", results_omp);
            }
        }

        // --- Benchmark da GPU (CUDA) ---
#ifdef WITH_CUDA
        if (gpu_benchmark) {
            auto tg0 = std::chrono::high_resolution_clock::now();
            results_gpu = gpu_processor->process_batch(sequence_subset);
            cudaDeviceSynchronize();
            auto tg1 = std::chrono::high_resolution_clock::now();
            time_gpu = std::chrono::duration<double>(tg1 - tg0).count();
            double sp_gpu = (time_cpu > 0) ? (time_cpu / time_gpu) : 0.0;
            std::cout << "-> GPU CUDA: " << time_gpu
                    << " s | Speedup=" << sp_gpu << std::endl;
            all_results.push_back({"GPU CUDA", num_seqs, -1, time_gpu, sp_gpu, -1.0});
            ReportGenerator::save_feature_matrix(
                "../data/gpu_results",
                results_gpu,
                feature_names,
                id_subset,
                "GPU_CUDA",
                num_seqs
            );

            if (cpu_sequential_benchmark) {
                check_matrix_consistency("CPU Seq", results_cpu, "GPU CUDA", results_gpu);
            }
        }
#endif
    }

    ReportGenerator::print_table(all_results);
    ReportGenerator::save_to_csv(all_results, "../data/benchmark_report.csv");
}


void BenchmarkRunner::run_k_variation_benchmark(const std::vector<int>& k_values, long long num_sequences, PseKNCParams base_params) {
    std::vector<BenchmarkResult> all_results;
    auto seq_pairs = std::vector<SequenceData>(main_sequences.begin(), main_sequences.begin() + num_sequences);
    std::vector<std::string> sequences;
    for(const auto& p : seq_pairs) sequences.push_back(p.second);

    for (int k : k_values) {
        std::cout << "\n===== Benchmark: K = " << k << " para " << num_sequences << " sequências =====" << std::endl;
        PseKNCParams current_params = base_params;
        current_params.k_value = k;
        
        // Processadores precisam ser recriados pois `sorted_ktuples` depende de K
        auto cpu_proc = std::make_unique<CPUProcessor>(properties, current_params);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (const auto& seq : sequences) {
            cpu_proc->process(seq);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double time_cpu = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "-> CPU Seq (K=" << k << "): " << time_cpu << " s" << std::endl;
        all_results.push_back({"CPU Seq", num_sequences, 1, k, current_params.lambda_max, time_cpu, 0,0});
    }
    ReportGenerator::print_table(all_results, true);
    ReportGenerator::save_to_csv(all_results, "benchmark_outputs/k_variation_report.csv");
}

// NOVO: Benchmark que varia o Lambda
void BenchmarkRunner::run_lambda_variation_benchmark(const std::vector<int>& lambda_values, const std::vector<long long>& num_sequences_to_run, PseKNCParams base_params) {
    std::vector<BenchmarkResult> all_results;
    auto cpu_proc = std::make_unique<CPUProcessor>(properties, base_params);

    for (long long num_seqs : num_sequences_to_run) {
        auto seq_pairs = std::vector<SequenceData>(main_sequences.begin(), main_sequences.begin() + num_seqs);
        std::vector<std::string> sequences;
        for(const auto& p : seq_pairs) sequences.push_back(p.second);

        for (int lambda : lambda_values) {
            std::cout << "\n===== Benchmark: Lambda = " << lambda << " para " << num_seqs << " sequências =====" << std::endl;
            PseKNCParams current_params = base_params;
            current_params.lambda_max = lambda;
            cpu_proc->params = current_params; // Atualiza os parâmetros do processador

            auto t0 = std::chrono::high_resolution_clock::now();
            for (const auto& seq : sequences) {
                cpu_proc->process(seq);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double time_cpu = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "-> CPU Seq (L=" << lambda << "): " << time_cpu << " s" << std::endl;
            all_results.push_back({"CPU Seq", num_seqs, 1, current_params.k_value, lambda, time_cpu, 0,0});
        }
    }
    ReportGenerator::print_table(all_results, true);
    ReportGenerator::save_to_csv(all_results, "benchmark_outputs/lambda_variation_report.csv");
}