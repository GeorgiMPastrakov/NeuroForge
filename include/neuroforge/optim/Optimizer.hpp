#pragma once

namespace neuroforge {

class Optimizer {
public:
    virtual void step() = 0;
    virtual void zeroGrad() = 0;
    virtual ~Optimizer() = default;
};

}
