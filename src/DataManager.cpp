#include "include/DataManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
DataManager::DataManager(std::string data_path, std::string sequence_path)
    : data_filepath(std::move(data_path)), sequence_filepath(std::move(sequence_path)) {}

PropertiesMap DataManager::load_properties() {
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

std::string DataManager::load_sequence() {
    std::ifstream file(sequence_filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Erro: Não foi possível abrir o arquivo de sequência: " + sequence_filepath);
    }
    
    std::string sequence;
    std::string line;
    while (std::getline(file, line)) {
        // Remove possíveis espaços em branco ou quebras de linha
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        sequence += line;
    }
    return sequence;
}