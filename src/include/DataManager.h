#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <string>
#include <vector>
#include <map>

// Mapa de um k-tuple (string) para o vetor de propriedades (doubles)
using PropertiesMap = std::map<std::string, std::vector<double>>;

class DataManager {
public:
    DataManager(std::string data_path, std::string sequence_path);
    PropertiesMap load_properties();
    std::string load_sequence();

private:
    std::string data_filepath;
    std::string sequence_filepath;
};

#endif // DATAMANAGER_H