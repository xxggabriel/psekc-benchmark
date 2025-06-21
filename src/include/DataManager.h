#ifndef DATAMANAGER_H
#define DATAMANAGER_H

#include <string>
#include <vector>
#include <map>

using PropertiesMap = std::map<std::string, std::vector<double>>;

class DataManager {
public:
    DataManager(std::string data_path, std::string sequence_path);
    bool setup_data_files();
    PropertiesMap load_and_augment_properties();
    std::vector<std::string> load_sequences();

private:
    bool _download_file(const std::string& url, const std::string& filename);
    std::string data_filepath;
    std::string sequence_filepath;
};

#endif // DATAMANAGER_H