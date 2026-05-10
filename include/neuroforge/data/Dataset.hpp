#pragma once

#include "neuroforge/core/Tensor.hpp"
#include "neuroforge/data/Batch.hpp"

#include <cstddef>
#include <vector>

namespace neuroforge {

class Dataset {
public:
    Dataset(Tensor features, Tensor labels);

    size_t size() const;
    const Tensor& features() const;
    const Tensor& labels() const;
    Batch sample(size_t index) const;
    Dataset subset(const std::vector<size_t>& indices) const;

private:
    Tensor features_;
    Tensor labels_;
};

}
