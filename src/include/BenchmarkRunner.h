#ifndef BENCHMARKRUNNER_H
#define BENCHMARKRUNNER_H

#include "PseKNCProcessor.h"
#include "ReportGenerator.h" // Inclui a definição de BenchmarkResult
#include <memory> // Para std::unique_ptr

class BenchmarkRunner {
public:
    BenchmarkRunner(std::string sequence, PseKNCParams params, const PropertiesMap& properties);
    void run(const std::vector<long long>& sizes);

private:
    std::string main_sequence;
    PseKNCParams params;
    const PropertiesMap& properties;

    std::unique_ptr<PseKNCProcessor> cpu_processor;
    std::unique_ptr<PseKNCProcessor> omp_processor;
#ifdef WITH_CUDA
    std::unique_ptr<PseKNCProcessor> gpu_processor;
#endif
};

#endif // BENCHMARKRUNNER_H