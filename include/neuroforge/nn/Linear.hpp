#pragma once

#include "neuroforge/nn/Module.hpp"
#include "neuroforge/nn/Parameter.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace neuroforge {

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Parameter*> parameters() override;
    std::string name() const override;

    Parameter& weights();
    const Parameter& weights() const;
    Parameter& bias();
    const Parameter& bias() const;
    size_t inFeatures() const;
    size_t outFeatures() const;
    size_t parameterCount() const;

private:
    size_t in_features_;
    size_t out_features_;
    Parameter weights_;
    Parameter bias_;
    std::optional<Tensor> cached_input_;

    void validateInput(const Tensor& input) const;
    void validateGradOutput(const Tensor& grad_output) const;
};

}
