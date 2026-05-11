#pragma once

#include "neuroforge/nn/Module.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace neuroforge {

class Dropout : public Module {
public:
    explicit Dropout(double probability, uint32_t seed = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override;

private:
    double probability_;
    uint32_t seed_;
    size_t forward_count_ = 0;
    std::optional<Tensor> cached_mask_;
};

}
