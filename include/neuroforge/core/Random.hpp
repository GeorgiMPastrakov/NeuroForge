#pragma once

#include "neuroforge/core/Shape.hpp"
#include "neuroforge/core/Tensor.hpp"

#include <cstdint>

namespace neuroforge {

class Random {
public:
    static void seed(uint32_t value);
    static Tensor uniform(const Shape& shape, double low, double high);
};

}
