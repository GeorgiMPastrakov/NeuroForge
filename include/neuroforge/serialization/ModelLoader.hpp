#pragma once

#include "neuroforge/nn/Sequential.hpp"

#include <string>

namespace neuroforge {

class ModelLoader {
public:
    static Sequential load(const std::string& path);
};

}
