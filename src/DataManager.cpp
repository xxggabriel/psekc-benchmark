#include "include/DataManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

DataManager::DataManager(std::string data_path, std::string sequence_path)
    : data_filepath(std::move(data_path)), sequence_filepath(std::move(sequence_path)) {}

PropertiesMap DataManager::load_and_augment_properties() {
    std::ifstream file(data_filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Erro: Não foi possível abrir o arquivo de propriedades: " + data_filepath);
    }

    PropertiesMap properties;
    std::string line;
    std::getline(file, line); // Ignora o cabeçalho

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string k_tuple;
        std::getline(ss, k_tuple, ',');

        std::vector<double> values;
        std::string value_str;
        while (std::getline(ss, value_str, ',')) {
            values.push_back(std::stod(value_str));
        }
        properties[k_tuple] = values;
    }
    std::cout << "Dicionário de propriedades carregado com " << properties.size() << " k-tuples." << std::endl;
    return properties;
}

std::vector<std::string> DataManager::load_sequences() {
    std::ifstream file(sequence_filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Erro: Não foi possível abrir o arquivo de sequência: " + sequence_filepath);
    }

    std::vector<std::string> sequences;
    std::string current_sequence;

    std::string line;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        if (line.empty()) continue;

        if (line[0] == '>') {
            if (!current_sequence.empty()) {
                sequences.push_back(current_sequence);
            }
            current_sequence.clear();
        } else {
            current_sequence += line;
        }
    }

    if (!current_sequence.empty()) {
        sequences.push_back(current_sequence);
    }

    if (sequences.empty()) {
        throw std::runtime_error("Nenhuma sequência encontrada no ficheiro FASTA.");
    }

    std::cout << "Carregadas " << sequences.size() << " sequências do ficheiro." << std::endl;
    return sequences;
}