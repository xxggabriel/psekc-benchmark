#ifndef BENCHMARKRUNNER_H
#define BENCHMARKRUNNER_H
#include "PseKNCProcessor.h"
#include "ReportGenerator.h"
#include <memory>

class BenchmarkRunner {
public:
    BenchmarkRunner(std::vector<SequenceData> sequences, PseKNCParams params, const PropertiesMap &properties);

    void run_scalability_benchmark(const std::vector<long long> &num_sequences_to_run);

    void run_k_variation_benchmark(const std::vector<int> &k_values, long long num_sequences, PseKNCParams base_params);

    void run_lambda_variation_benchmark(const std::vector<int> &lambda_values,
                                        const std::vector<long long> &num_sequences_to_run, PseKNCParams base_params);

private:
    std::vector<SequenceData> main_sequences;
    PseKNCParams params;
    const PropertiesMap &properties;
    std::unique_ptr<PseKNCProcessor> cpu_processor;
    std::unique_ptr<PseKNCProcessor> omp_processor;
#ifdef WITH_CUDA
    std::unique_ptr<PseKNCProcessor> gpu_processor;
#endif
};
#endif // BENCHMARKRUNNER_H
