#include "neuroforge/autograd/Value.hpp"

#include <cassert>
#include <cmath>
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
    Value a(2.0, true);
    Value b(3.0, true);
    Value c = a * b;
    Value d = c + a;
    d.backward();
    assert(near(d.data(), 8.0));
    assert(near(a.grad(), 4.0));
    assert(near(b.grad(), 2.0));

    Value x(3.0, true);
    Value y = x * x + x;
    y.backward();
    assert(near(x.grad(), 7.0));
    x.zeroGrad();
    assert(near(x.grad(), 0.0));

    Value p(4.0, true);
    Value q(2.0, true);
    Value r = p / q - q;
    r.backward();
    assert(near(p.grad(), 0.5));
    assert(near(q.grad(), -2.0));

    Value pow_input(3.0, true);
    Value pow_output = pow_input.pow(3.0);
    pow_output.backward();
    assert(near(pow_output.data(), 27.0));
    assert(near(pow_input.grad(), 27.0));

    Value relu_negative(-2.0, true);
    Value relu_output = relu_negative.relu();
    relu_output.backward();
    assert(near(relu_output.data(), 0.0));
    assert(near(relu_negative.grad(), 0.0));

    Value sigmoid_input(0.0, true);
    Value sigmoid_output = sigmoid_input.sigmoid();
    sigmoid_output.backward();
    assert(near(sigmoid_output.data(), 0.5));
    assert(near(sigmoid_input.grad(), 0.25));

    Value tanh_input(0.0, true);
    Value tanh_output = tanh_input.tanh();
    tanh_output.backward();
    assert(near(tanh_output.data(), 0.0));
    assert(near(tanh_input.grad(), 1.0));

    expectThrows<std::invalid_argument>([] {
        Value one(1.0, true);
        Value zero(0.0, true);
        Value result = one / zero;
        result.backward();
    });

    return 0;
}
