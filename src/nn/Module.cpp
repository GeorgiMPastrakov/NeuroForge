#include "neuroforge/nn/Module.hpp"

namespace neuroforge {

std::vector<Parameter*> Module::parameters() {
    return {};
}

void Module::zero_grad() {
    for (Parameter* parameter : parameters()) {
        parameter->zero_grad();
    }
}

}
