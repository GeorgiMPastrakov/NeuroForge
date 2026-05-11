#include "neuroforge/serialization/ModelLoader.hpp"

#include "neuroforge/nn/Dropout.hpp"
#include "neuroforge/nn/LeakyReLU.hpp"
#include "neuroforge/nn/Linear.hpp"
#include "neuroforge/nn/ReLU.hpp"
#include "neuroforge/nn/Sigmoid.hpp"
#include "neuroforge/nn/Softmax.hpp"
#include "neuroforge/nn/Tanh.hpp"

#include <fstream>
#include <memory>
#include <stdexcept>

namespace neuroforge {

namespace {

void requireToken(const std::string& actual, const std::string& expected) {
    if (actual != expected) {
        throw std::invalid_argument("ModelLoader expected token " + expected + ".");
    }
}

std::vector<double> readValues(std::ifstream& file, size_t expected) {
    size_t count = 0;
    file >> count;

    if (!file || count != expected) {
        throw std::invalid_argument("ModelLoader tensor value count mismatch.");
    }

    std::vector<double> values(count);

    for (size_t index = 0; index < count; ++index) {
        file >> values[index];

        if (!file) {
            throw std::invalid_argument("ModelLoader could not read tensor value.");
        }
    }

    return values;
}

}

Sequential ModelLoader::load(const std::string& path) {
    std::ifstream file(path);

    if (!file) {
        throw std::invalid_argument("ModelLoader could not open file: " + path);
    }

    std::string token;
    size_t version = 0;
    file >> token >> version;
    requireToken(token, "NeuroForgeSequential");

    if (version != 1) {
        throw std::invalid_argument("ModelLoader unsupported model version.");
    }

    size_t layer_count = 0;
    file >> token >> layer_count;
    requireToken(token, "Layers");

    Sequential model;

    for (size_t layer = 0; layer < layer_count; ++layer) {
        file >> token;

        if (!file) {
            throw std::invalid_argument("ModelLoader missing layer token.");
        }

        if (token == "Linear") {
            size_t in_features = 0;
            size_t out_features = 0;
            file >> in_features >> out_features;

            if (!file || in_features == 0 || out_features == 0) {
                throw std::invalid_argument("ModelLoader invalid Linear dimensions.");
            }

            auto linear = std::make_shared<Linear>(in_features, out_features);

            file >> token;
            requireToken(token, "Weights");
            std::vector<double> weights = readValues(file, in_features * out_features);

            file >> token;
            requireToken(token, "Bias");
            std::vector<double> bias = readValues(file, out_features);

            for (size_t index = 0; index < weights.size(); ++index) {
                linear->weights().data().data()[index] = weights[index];
            }

            for (size_t index = 0; index < bias.size(); ++index) {
                linear->bias().data().data()[index] = bias[index];
            }

            model.add(linear);
        } else if (token == "ReLU") {
            model.add(std::make_shared<ReLU>());
        } else if (token == "Sigmoid") {
            model.add(std::make_shared<Sigmoid>());
        } else if (token == "Tanh") {
            model.add(std::make_shared<Tanh>());
        } else if (token == "LeakyReLU") {
            double negative_slope = 0.0;
            file >> negative_slope;

            if (!file) {
                throw std::invalid_argument("ModelLoader invalid LeakyReLU slope.");
            }

            model.add(std::make_shared<LeakyReLU>(negative_slope));
        } else if (token == "Softmax") {
            model.add(std::make_shared<Softmax>());
        } else if (token == "Dropout") {
            double probability = 0.0;
            uint32_t seed = 0;
            file >> probability >> seed;

            if (!file) {
                throw std::invalid_argument("ModelLoader invalid Dropout parameters.");
            }

            model.add(std::make_shared<Dropout>(probability, seed));
        } else {
            throw std::invalid_argument("ModelLoader unsupported layer token: " + token);
        }
    }

    return model;
}

}
