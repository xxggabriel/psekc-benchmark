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

    static void save_feature_matrix(
        std::string base_dir,
        const std::vector<std::vector<double>>& matrix,
        const std::vector<std::string>& feature_names,
        const std::vector<std::string>& sequence_ids,
        const std::string& platform_name,
        long long num_sequences
    );
};

#endif // REPORTGENERATOR_H