#pragma once

#include "neuroforge/nn/Sequential.hpp"

#include <string>

namespace neuroforge {

class ModelSaver {
public:
    static void save(const Sequential& model, const std::string& path);
};

}
