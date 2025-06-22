#include "include/DataManager.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <curl/curl.h>
#include <filesystem>
#include <string>

static size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

DataManager::DataManager(std::string d_url, std::string d_filename,
                         std::string s_url, std::string s_filename)
    : data_url(std::move(d_url)), data_filename(std::move(d_filename)),
      sequence_url(std::move(s_url)), sequence_filename(std::move(s_filename)) {
}

bool DataManager::_download_file(const std::string &url, const std::string &filename) {
    if (std::filesystem::exists(filename)) {
        return true;
    }

    CURL *curl_handle;
    FILE *pagefile;

    curl_global_init(CURL_GLOBAL_ALL);
    curl_handle = curl_easy_init();

    if (curl_handle) {
        curl_easy_setopt(curl_handle, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_handle, CURLOPT_VERBOSE, 0L); // Desativa logs do curl
        curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 1L);
        curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, write_data);

        pagefile = fopen(filename.c_str(), "wb");
        if (pagefile) {
            curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, pagefile);
            CURLcode res = curl_easy_perform(curl_handle);
            fclose(pagefile);
            if (res != CURLE_OK) {
                std::cerr << "Erro no download: " << curl_easy_strerror(res) << std::endl;
                curl_easy_cleanup(curl_handle);
                curl_global_cleanup();
                return false;
            }
        } else {
            std::cerr << "Erro: Não foi possível abrir o ficheiro para escrita: " << filename << std::endl;
            curl_easy_cleanup(curl_handle);
            curl_global_cleanup();
            return false;
        }
        curl_easy_cleanup(curl_handle);
    }
    curl_global_cleanup();
    std::cout << "Ficheiro '" << filename << "' baixado com sucesso." << std::endl;
    return true;
}


bool DataManager::setup_data_files() {
    std::cout << "Verificando ficheiros de dados..." << std::endl;
    std::filesystem::create_directories("data"); // Garante que a pasta 'data' existe
    bool data_ok = _download_file(data_url, "data/" + data_filename);
    bool seq_ok = _download_file(sequence_url, "data/" + sequence_filename);
    return data_ok && seq_ok;
}


PropertiesMap DataManager::load_and_augment_properties() {
    std::string filepath = data_filename; // cuide do caminho relativo correto
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Erro: Não foi possível abrir o arquivo de propriedades: " + filepath);
    }

    auto trim = [](std::string &s) {
        const char* ws = " \t\n\r";
        size_t start = s.find_first_not_of(ws);
        size_t end   = s.find_last_not_of(ws);
        if (start == std::string::npos) { s.clear(); }
        else { s = s.substr(start, end - start + 1); }
    };

    PropertiesMap properties;
    std::string line;
    std::getline(file, line); // Ignora o cabeçalho
    size_t line_num = 1;

    while (std::getline(file, line)) {
        ++line_num;
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string k_tuple;
        if (!std::getline(ss, k_tuple, ',')) continue;
        trim(k_tuple);

        std::vector<double> values;
        std::string token;
        while (std::getline(ss, token, ',')) {
            trim(token);
            if (token.empty()) continue;
            try {
                values.push_back(std::stod(token));
            } catch (const std::invalid_argument&) {
                std::cerr << "std::stod falhou ao converter '"
                          << token
                          << "' na linha " << line_num
                          << " do arquivo " << filepath << std::endl;
                throw;
            }
        }

        if (!values.empty()) {
            properties.emplace(k_tuple, std::move(values));
        }
    }

    std::cout << "Dicionário de propriedades carregado com "
              << properties.size()
              << " k-tuples." << std::endl;
    return properties;
}


std::vector<SequenceData> DataManager::load_sequences() {
    std::string filepath = "../data/" + sequence_filename;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Erro: Não foi possível abrir o arquivo de sequência: " + filepath);
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
