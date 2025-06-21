#ifndef BENCHMARKRUNNER_H
#define BENCHMARKRUNNER_H

#include "PseKNCProcessor.h"
#include <memory>
#include <vector>
#ifdef WITH_CUDA
#include "include/GPUProcessor.h"
#endif
#include "include/OMPProcessor.h"


class BenchmarkRunner {
public:
    BenchmarkRunner(std::vector<SequenceData> sequences,
                    PseKNCParams params,
                    const PropertiesMap &properties);

    void run(const std::vector<long long> &num_sequences_to_run);

private:
    std::vector<SequenceData> main_sequences;
    PseKNCParams params;
    const PropertiesMap &properties;
    std::unique_ptr<PseKNCProcessor> cpu_processor;
    std::unique_ptr<OMPProcessor> omp_processor;
#ifdef WITH_CUDA
    std::unique_ptr<GPUProcessor> gpu_processor;
#endif
};

#endif // BENCHMARKRUNNER_H
