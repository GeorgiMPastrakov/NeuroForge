#include "neuroforge/core/Random.hpp"

#include <random>
#include <stdexcept>
#include <vector>

namespace neuroforge {

namespace {

std::mt19937& engine() {
    static std::mt19937 generator(0);
    return generator;
}

}

void Random::seed(uint32_t value) {
    engine().seed(value);
}

Tensor Random::uniform(const Shape& shape, double low, double high) {
    if (low > high) {
        throw std::invalid_argument("Random uniform lower bound must be less than or equal to upper bound.");
    }

    std::uniform_real_distribution<double> distribution(low, high);
    std::vector<double> data;
    data.reserve(shape.size());

    for (size_t index = 0; index < shape.size(); ++index) {
        data.push_back(distribution(engine()));
    }

    return Tensor(std::move(data), shape);
}

}
