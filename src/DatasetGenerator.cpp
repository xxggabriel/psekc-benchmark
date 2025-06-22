#include "include/DatasetGenerator.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <string>
DatasetGenerator::DatasetGenerator() {
    std::random_device rd;
    random_engine.seed(rd());
}

std::vector<char> get_alphabet_chars(AlphabetType type) {
    if (type == AlphabetType::RNA) {
        return {'A', 'C', 'G', 'U'};
    }
    return {'A', 'C', 'G', 'T'};
}

void DatasetGenerator::generate_ktuples_recursive(
    int k,
    const std::vector<char>& alphabet,
    std::string current_ktuple,
    std::vector<std::string>& all_ktuples) const
{
    if (current_ktuple.length() == k) {
        all_ktuples.push_back(current_ktuple);
        return;
    }

    for (char nucleotide : alphabet) {
        generate_ktuples_recursive(k, alphabet, current_ktuple + nucleotide, all_ktuples);
    }
}

void DatasetGenerator::generate_properties_file(
    int k,
    int num_properties,
    AlphabetType type,
    const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Erro: Não foi possível criar o ficheiro de propriedades: " + filename);
    }

    std::cout << "Gerando ficheiro de propriedades '" << filename << "'..." << std::endl;

    file << "k_tuple";
    for (int i = 1; i <= num_properties; ++i) {
        file << ",prop_" << i;
    }
    file << "\n";

    std::vector<std::string> ktuples_to_generate;
    if (type == AlphabetType::DNA_AND_RNA) {
        generate_ktuples_recursive(k, {'A', 'C', 'G', 'T'}, "", ktuples_to_generate);
        generate_ktuples_recursive(k, {'A', 'C', 'G', 'U'}, "", ktuples_to_generate);
    } else {
        generate_ktuples_recursive(k, get_alphabet_chars(type), "", ktuples_to_generate);
    }

    std::sort(ktuples_to_generate.begin(), ktuples_to_generate.end());
    ktuples_to_generate.erase(std::unique(ktuples_to_generate.begin(), ktuples_to_generate.end()), ktuples_to_generate.end());

    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    file << std::fixed << std::setprecision(4);
    for (const auto& kt : ktuples_to_generate) {
        file << kt;
        for (int i = 0; i < num_properties; ++i) {
            file << "," << dist(random_engine);
        }
        file << "\n";
    }

    std::cout << "Ficheiro de propriedades com " << ktuples_to_generate.size() << " k-tuples gerado com sucesso." << std::endl;
}

std::string DatasetGenerator::generate_random_sequence(
    int length,
    const std::vector<char>& alphabet) const
{
    std::uniform_int_distribution<int> dist(0, alphabet.size() - 1);
    std::string sequence;
    sequence.reserve(length);
    for (int i = 0; i < length; ++i) {
        sequence += alphabet[dist(random_engine)];
    }
    return sequence;
}

void DatasetGenerator::generate_fasta_file(
    int num_sequences,
    int min_len,
    int max_len,
    AlphabetType type,
    const std::string& filename) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Erro: Não foi possível criar o ficheiro FASTA: " + filename);
    }
    if (min_len > max_len) {
        throw std::invalid_argument("O comprimento mínimo da sequência não pode ser maior que o máximo.");
    }

    std::cout << "Gerando ficheiro FASTA '" << filename << "'..." << std::endl;

    auto alphabet = get_alphabet_chars(type);
    std::uniform_int_distribution<int> len_dist(min_len, max_len);
    const int line_width = 80;

    for (int i = 1; i <= num_sequences; ++i) {
        
        file << ">seq_" << i << "_len_" << (max_len - min_len) << "\n";
        
        
        int length = len_dist(random_engine);
        std::string sequence = generate_random_sequence(length, alphabet);

        
        for (int j = 0; j < length; j += line_width) {
            file << sequence.substr(j, line_width) << "\n";
        }
    }
    std::cout << num_sequences << " sequências geradas com sucesso." << std::endl;
}
