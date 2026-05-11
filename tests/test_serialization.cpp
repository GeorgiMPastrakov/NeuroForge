#include "neuroforge/serialization/ModelLoader.hpp"
#include "neuroforge/serialization/ModelSaver.hpp"
#include "neuroforge/nn/Dropout.hpp"
#include "neuroforge/nn/LeakyReLU.hpp"
#include "neuroforge/nn/Linear.hpp"
#include "neuroforge/nn/ReLU.hpp"
#include "neuroforge/nn/Sequential.hpp"
#include "neuroforge/nn/Sigmoid.hpp"
#include "neuroforge/nn/Softmax.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

using namespace neuroforge;

template <typename Exception, typename Function>
void expectThrows(Function function) {
    bool thrown = false;

    try {
        function();
    } catch (const Exception&) {
        thrown = true;
    }

    assert(thrown);
}

bool near(double left, double right) {
    return std::fabs(left - right) < 1e-9;
}

int main() {
    Sequential model;
    auto linear1 = std::make_shared<Linear>(2, 2);
    linear1->weights().data().at(0, 0) = 1.0;
    linear1->weights().data().at(0, 1) = -1.0;
    linear1->weights().data().at(1, 0) = 0.5;
    linear1->weights().data().at(1, 1) = 2.0;
    linear1->bias().data().at(0, 0) = 0.25;
    linear1->bias().data().at(0, 1) = -0.5;
    auto linear2 = std::make_shared<Linear>(2, 1);
    linear2->weights().data().at(0, 0) = 1.5;
    linear2->weights().data().at(1, 0) = -0.25;
    linear2->bias().data().at(0, 0) = 0.75;

    model.add(linear1);
    model.add(std::make_shared<ReLU>());
    model.add(linear2);
    model.add(std::make_shared<Sigmoid>());

    Tensor input = Tensor::fromVector({
        {1.0, 2.0},
        {3.0, 4.0}
    });
    Tensor before = model.forward(input);

    std::filesystem::path path = std::filesystem::temp_directory_path() / "neuroforge_model.txt";
    ModelSaver::save(model, path.string());
    Sequential loaded = ModelLoader::load(path.string());
    Tensor after = loaded.forward(input);

    assert(loaded.size() == 4);
    assert(loaded.summary().find("Total parameters: 9") != std::string::npos);

    for (size_t index = 0; index < before.size(); ++index) {
        assert(near(before.data()[index], after.data()[index]));
    }

    expectThrows<std::invalid_argument>([] {
        ModelLoader::load("missing_neuroforge_model.txt");
    });

    std::filesystem::path bad_path = std::filesystem::temp_directory_path() / "neuroforge_bad_model.txt";
    {
        std::ofstream file(bad_path);
        file << "BadModel 1\n";
    }
    expectThrows<std::invalid_argument>([&] {
        ModelLoader::load(bad_path.string());
    });

    std::filesystem::path malformed_path = std::filesystem::temp_directory_path() / "neuroforge_malformed_model.txt";
    {
        std::ofstream file(malformed_path);
        file << "NeuroForgeSequential 1\nLayers 1\nLinear 2 1\nWeights 1 0.0\nBias 1 0.0\n";
    }
    expectThrows<std::invalid_argument>([&] {
        ModelLoader::load(malformed_path.string());
    });

    Sequential phase9_model;
    phase9_model.add(std::make_shared<Linear>(2, 3));
    phase9_model.add(std::make_shared<LeakyReLU>(0.2));
    phase9_model.add(std::make_shared<Dropout>(0.25, 17));
    phase9_model.add(std::make_shared<Linear>(3, 2));
    phase9_model.add(std::make_shared<Softmax>());
    phase9_model.eval();

    Tensor phase9_input = Tensor::fromVector({
        {0.2, -0.4},
        {1.0, 0.5}
    });
    Tensor phase9_before = phase9_model.forward(phase9_input);

    std::filesystem::path phase9_path = std::filesystem::temp_directory_path() / "neuroforge_phase9_model.txt";
    ModelSaver::save(phase9_model, phase9_path.string());
    Sequential phase9_loaded = ModelLoader::load(phase9_path.string());
    phase9_loaded.eval();
    Tensor phase9_after = phase9_loaded.forward(phase9_input);

    assert(phase9_loaded.size() == 5);

    for (size_t index = 0; index < phase9_before.size(); ++index) {
        assert(near(phase9_before.data()[index], phase9_after.data()[index]));
    }

    return 0;
}
