#ifndef DATASETGENERATOR_H
#define DATASETGENERATOR_H

#include <string>
#include <vector>
#include <random>

enum class AlphabetType {
    RNA,
    DNA,
    DNA_AND_RNA
};

class DatasetGenerator {
public:
    DatasetGenerator();

    void generate_properties_file(
        int k,
        int num_properties,
        AlphabetType type,
        const std::string& filename
    ) const;

    void generate_fasta_file(
        int num_sequences,
        int min_len,
        int max_len,
        AlphabetType type,
        const std::string& filename
    ) const;

private:
    mutable std::mt19937 random_engine;
    
    
    std::vector<std::string> get_alphabet(AlphabetType type) const;
    void generate_ktuples_recursive(
        int k,
        const std::vector<char>& alphabet,
        std::string current_ktuple,
        std::vector<std::string>& all_ktuples
    ) const;
    std::string generate_random_sequence(
        int length,
        const std::vector<char>& alphabet
    ) const;
};

#endif // DATASETGENERATOR_H