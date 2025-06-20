#include "include/BenchmarkRunner.h"
#include "include/CPUProcessor.h"
#include "include/OMPProcessor.h"
#ifdef WITH_CUDA
#include "include/GPUProcessor.h"
#endif
#include <chrono>
#include <omp.h>

BenchmarkRunner::BenchmarkRunner(std::string sequence, PseKNCParams params, const PropertiesMap& properties)
    : main_sequence(std::move(sequence)), params(params), properties(properties) {
    cpu_processor = std::make_unique<CPUProcessor>(properties, params);
    omp_processor = std::make_unique<OMPProcessor>(properties, params);
    #ifdef WITH_CUDA
    gpu_processor = std::make_unique<GPUProcessor>(properties, params);
    #endif
}

void BenchmarkRunner::run(const std::vector<long long>& sizes) {
    std::vector<BenchmarkResult> all_results;

    for (long long size : sizes) {
        if (size > main_sequence.length()) {
            std::cout << "\nAviso: Tamanho de sequência solicitado (" << size << ") é maior que o disponível (" 
                      << main_sequence.length() << "). A ignorar este tamanho." << std::endl;
            continue;
        }
        
        std::string sub_sequence = main_sequence.substr(0, size);

        std::cout << "\n======================================================================" << std::endl;
        std::cout << "Iniciando benchmark para tamanho de sequência: " << size << " bases" << std::endl;
        std::cout << "======================================================================" << std::endl;




        // --- Benchmark da GPU (CUDA) ---
        double duration_gpu_val = -1.0;
        #ifdef WITH_CUDA
            auto start_gpu = std::chrono::high_resolution_clock::now();
            auto vec_gpu = gpu_processor->process(sub_sequence);
            #ifdef __CUDACC__
            cudaDeviceSynchronize();
            #endif
            auto end_gpu = std::chrono::high_resolution_clock::now();
            duration_gpu_val = std::chrono::duration<double>(end_gpu - start_gpu).count();
            std::cout << "-> GPU (CUDA) Concluído em: " << duration_gpu_val << "s" << std::endl;
        #endif

        // --- Benchmark da CPU Paralela (OpenMP) ---
        int num_threads = omp_get_max_threads();
        auto start_omp = std::chrono::high_resolution_clock::now();
        auto vec_omp = omp_processor->process(sub_sequence);
        auto end_omp = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_omp = end_omp - start_omp;
        std::cout << "-> CPU Paralelo (OpenMP) Concluído em: " << duration_omp.count() << "s" << std::endl;

        // --- Benchmark da CPU Sequencial ---
        auto start_cpu = std::chrono::high_resolution_clock::now();
        auto vec_cpu = cpu_processor->process(sub_sequence);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_cpu = end_cpu - start_cpu;
        std::cout << "-> CPU Sequencial Concluído em: " << duration_cpu.count() << "s" << std::endl;


        // --- Armazenar Resultados ---
        double speedup_cpu = 1.0;
        double speedup_omp = duration_cpu.count() / duration_omp.count();
        double efficiency_omp = (speedup_omp / num_threads) * 100.0;
        
        all_results.push_back({"CPU Sequencial", size, 1, duration_cpu.count(), speedup_cpu, 100.0});
        all_results.push_back({"CPU Paralelo (OMP)", size, num_threads, duration_omp.count(), speedup_omp, efficiency_omp});

        if (duration_gpu_val >= 0) {
            double speedup_gpu = duration_cpu.count() / duration_gpu_val;
            all_results.push_back({"GPU (CUDA)", size, -1, duration_gpu_val, speedup_gpu, -1.0});
        }
    }

    // --- Gerar Relatórios ---
    ReportGenerator::print_table(all_results);
    ReportGenerator::save_to_csv(all_results, "benchmark_report.csv");
}