#pragma once

#include "neuroforge/nn/Module.hpp"

#include <memory>
#include <string>
#include <vector>

namespace neuroforge {

class Sequential : public Module {
public:
    void add(std::shared_ptr<Module> module);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::vector<Parameter*> parameters() override;
    std::string name() const override;
    std::string summary() const;
    size_t size() const;
    const std::vector<std::shared_ptr<Module>>& layers() const;

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

}
