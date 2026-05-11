#include "neuroforge/serialization/ModelSaver.hpp"

#include "neuroforge/nn/Dropout.hpp"
#include "neuroforge/nn/LeakyReLU.hpp"
#include "neuroforge/nn/Linear.hpp"
#include "neuroforge/nn/ReLU.hpp"
#include "neuroforge/nn/Sigmoid.hpp"
#include "neuroforge/nn/Softmax.hpp"
#include "neuroforge/nn/Tanh.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace neuroforge {

namespace {

void writeTensor(std::ofstream& file, const Tensor& tensor) {
    file << tensor.size();

    for (double value : tensor.data()) {
        file << " " << std::setprecision(17) << value;
    }

    file << "\n";
}

}

void ModelSaver::save(const Sequential& model, const std::string& path) {
    std::ofstream file(path);

    if (!file) {
        throw std::invalid_argument("ModelSaver could not open file: " + path);
    }

    file << "NeuroForgeSequential 1\n";
    file << "Layers " << model.layers().size() << "\n";

    for (const auto& module : model.layers()) {
        if (const auto* linear = dynamic_cast<const Linear*>(module.get())) {
            file << "Linear " << linear->inFeatures() << " " << linear->outFeatures() << "\n";
            file << "Weights ";
            writeTensor(file, linear->weights().data());
            file << "Bias ";
            writeTensor(file, linear->bias().data());
        } else if (dynamic_cast<const ReLU*>(module.get())) {
            file << "ReLU\n";
        } else if (dynamic_cast<const Sigmoid*>(module.get())) {
            file << "Sigmoid\n";
        } else if (dynamic_cast<const Tanh*>(module.get())) {
            file << "Tanh\n";
        } else if (const auto* leaky_relu = dynamic_cast<const LeakyReLU*>(module.get())) {
            file << "LeakyReLU " << std::setprecision(17) << leaky_relu->negativeSlope() << "\n";
        } else if (dynamic_cast<const Softmax*>(module.get())) {
            file << "Softmax\n";
        } else if (const auto* dropout = dynamic_cast<const Dropout*>(module.get())) {
            file << "Dropout " << std::setprecision(17) << dropout->probability() << " " << dropout->seed() << "\n";
        } else {
            throw std::invalid_argument("ModelSaver encountered unsupported layer: " + module->name());
        }
    }
}

}
