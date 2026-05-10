#pragma once

#include "neuroforge/data/Batch.hpp"
#include "neuroforge/data/Dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace neuroforge {

class DataLoader {
public:
    DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle = false, uint32_t seed = 0);

    std::vector<Batch> batches() const;

private:
    const Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    uint32_t seed_;
};

}
