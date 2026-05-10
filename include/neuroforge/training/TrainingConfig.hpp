#pragma once

#include <cstddef>

namespace neuroforge {

struct TrainingConfig {
    size_t epochs = 1000;
    size_t log_interval = 100;
    bool verbose = false;
};

}
