#ifndef REPORTGENERATOR_H
#define REPORTGENERATOR_H

#include <string>
#include <vector>
#include <iostream>

struct BenchmarkResult {
    std::string platform;
    long long sequence_size;
    int num_threads;
    double duration_sec;
    double speedup;
    double efficiency_percent;
};

class ReportGenerator {
public:
    static void print_table(const std::vector<BenchmarkResult>& results);
    static void save_to_csv(const std::vector<BenchmarkResult>& results, const std::string& filename);
};

#endif // REPORTGENERATOR_H