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

std::vector<SequenceData> DataManager::load_sequences() {
    std::ifstream file(sequence_filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Erro: Não foi possível abrir o arquivo de sequência: " + sequence_filepath);
    }

    std::vector<SequenceData> sequences;
    std::string current_sequence;
    std::string current_id;

    std::string line;
    while (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        if (line.empty()) continue;

        if (line[0] == '>') {
            // Se já tínhamos uma sequência e um ID, guarda o par
            if (!current_sequence.empty() && !current_id.empty()) {
                sequences.push_back({current_id, current_sequence});
            }
            // Guarda o novo ID (sem o '>') e limpa a sequência atual
            current_id = line.substr(1);
            current_sequence.clear();
        } else {
            current_sequence += line;
        }
    }

    // Garante o push da ultima sequência
    if (!current_sequence.empty() && !current_id.empty()) {
        sequences.push_back({current_id, current_sequence});
    }

    if (sequences.empty()) {
        throw std::runtime_error("Nenhuma sequência encontrada no ficheiro FASTA.");
    }

    std::cout << "Carregadas " << sequences.size() << " sequências do ficheiro." << std::endl;
    return sequences;
}