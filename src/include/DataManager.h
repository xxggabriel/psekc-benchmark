#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <string>
#include <vector>
#include <map>
#include <utility>

using PropertiesMap = std::map<std::string, std::vector<double> >;
using SequenceData = std::pair<std::string, std::string>; // <ID, Sequência>

class DataManager {
public:
    DataManager(std::string data_url, std::string data_filename,
                std::string sequence_url, std::string sequence_filename);

    bool setup_data_files();

    PropertiesMap load_and_augment_properties();

    std::vector<SequenceData> load_sequences();

private:
    bool _download_file(const std::string &url, const std::string &filename);

    std::string data_url;
    std::string data_filename;
    std::string sequence_url;
    std::string sequence_filename;
};

#endif // DATAMANAGER_H
