#ifndef BENCHMARKRUNNER_H
#define BENCHMARKRUNNER_H
#include "PseKNCProcessor.h"
#include "ReportGenerator.h"
#include <memory>

class BenchmarkRunner {
public:
    BenchmarkRunner(std::vector<std::string> sequences, PseKNCParams params, const PropertiesMap& properties);
    void run(const std::vector<long long>& num_sequences_to_run);
private:
    std::vector<std::string> main_sequences;
    PseKNCParams params;
    const PropertiesMap& properties;
    std::unique_ptr<PseKNCProcessor> cpu_processor;
    std::unique_ptr<PseKNCProcessor> omp_processor;
#ifdef WITH_CUDA
    std::unique_ptr<PseKNCProcessor> gpu_processor;
#endif
};
#endif // BENCHMARKRUNNER_H