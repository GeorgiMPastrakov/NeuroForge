#pragma once

#include "neuroforge/core/Tensor.hpp"

namespace neuroforge {

struct Batch {
    Tensor features;
    Tensor labels;
};

}
