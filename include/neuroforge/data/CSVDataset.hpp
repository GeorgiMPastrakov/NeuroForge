#pragma once

#include "neuroforge/data/Dataset.hpp"

#include <cstddef>
#include <string>

namespace neuroforge {

class CSVDataset : public Dataset {
public:
    CSVDataset(const std::string& path, size_t label_column, bool has_header = true);

private:
    static Dataset load(const std::string& path, size_t label_column, bool has_header);
};

}
