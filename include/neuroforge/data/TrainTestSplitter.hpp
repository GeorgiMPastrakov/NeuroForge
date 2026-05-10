#pragma once

#include "neuroforge/data/Dataset.hpp"

#include <cstdint>

namespace neuroforge {

struct TrainTestSplit {
    Dataset train;
    Dataset test;
};

class TrainTestSplitter {
public:
    static TrainTestSplit split(const Dataset& dataset, double train_ratio, bool shuffle = true, uint32_t seed = 0);
};

}
