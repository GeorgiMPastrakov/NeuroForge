#include "neuroforge/nn/Sequential.hpp"

#include <sstream>
#include <stdexcept>

namespace neuroforge {

void Sequential::add(std::shared_ptr<Module> module) {
    if (!module) {
        throw std::invalid_argument("Sequential cannot add a null module.");
    }

    modules_.push_back(std::move(module));
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor output = input;

    for (const auto& module : modules_) {
        output = module->forward(output);
    }

    return output;
}

Tensor Sequential::backward(const Tensor& grad_output) {
    Tensor grad = grad_output;

    for (auto iterator = modules_.rbegin(); iterator != modules_.rend(); ++iterator) {
        grad = (*iterator)->backward(grad);
    }

    return grad;
}

std::vector<Parameter*> Sequential::parameters() {
    std::vector<Parameter*> result;

    for (const auto& module : modules_) {
        std::vector<Parameter*> child_parameters = module->parameters();
        result.insert(result.end(), child_parameters.begin(), child_parameters.end());
    }

    return result;
}

std::string Sequential::name() const {
    return "Sequential";
}

std::string Sequential::summary() const {
    std::ostringstream stream;
    size_t total = 0;

    stream << "Sequential(\n";

    for (size_t index = 0; index < modules_.size(); ++index) {
        size_t count = 0;

        for (Parameter* parameter : modules_[index]->parameters()) {
            count += parameter->data().size();
        }

        total += count;
        stream << "  [" << index << "] " << modules_[index]->name();

        if (count > 0) {
            stream << ", parameters: " << count;
        }

        stream << "\n";
    }

    stream << ")\n\nTotal parameters: " << total;
    return stream.str();
}

size_t Sequential::size() const {
    return modules_.size();
}

const std::vector<std::shared_ptr<Module>>& Sequential::layers() const {
    return modules_;
}

}
