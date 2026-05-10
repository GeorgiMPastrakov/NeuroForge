#include "neuroforge/autograd/Value.hpp"
#include "neuroforge/core/Tensor.hpp"

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
    y.backward();
    assert(near(x.grad(), 14.0));
    x.zeroGrad();
    assert(near(x.grad(), 0.0));
    y.backward();
    assert(near(x.grad(), 7.0));

    Tensor scalar(std::vector<double>{2.0}, Shape({1}), true);
    assert(scalar.requiresGrad());
    assert(scalar.grad().size() == 1);
    assert(near(scalar.grad()[0], 0.0));
    scalar.backward();
    assert(near(scalar.grad()[0], 1.0));
    scalar.backward();
    assert(near(scalar.grad()[0], 2.0));
    scalar.zeroGrad();
    assert(near(scalar.grad()[0], 0.0));

    Tensor metadata = Tensor::fromVector({1.0, 2.0});
    assert(!metadata.requiresGrad());
    metadata.setRequiresGrad(true);
    assert(metadata.requiresGrad());
    assert(metadata.grad().size() == metadata.size());
    metadata.grad()[0] = 3.0;
    metadata.zeroGrad();
    assert(near(metadata.grad()[0], 0.0));

    expectThrows<std::invalid_argument>([] {
        Tensor vector = Tensor::fromVector({1.0, 2.0});
        vector.backward();
    });

    Tensor add_left(std::vector<double>{1.0, 2.0}, Shape({2}), true);
    Tensor add_right(std::vector<double>{3.0, 4.0}, Shape({2}), true);
    add_left.add(add_right).sum().backward();
    assert(near(add_left.grad()[0], 1.0));
    assert(near(add_left.grad()[1], 1.0));
    assert(near(add_right.grad()[0], 1.0));
    assert(near(add_right.grad()[1], 1.0));

    Tensor subtract_left(std::vector<double>{1.0, 2.0}, Shape({2}), true);
    Tensor subtract_right(std::vector<double>{3.0, 4.0}, Shape({2}), true);
    subtract_left.subtract(subtract_right).sum().backward();
    assert(near(subtract_left.grad()[0], 1.0));
    assert(near(subtract_left.grad()[1], 1.0));
    assert(near(subtract_right.grad()[0], -1.0));
    assert(near(subtract_right.grad()[1], -1.0));

    Tensor multiply_left(std::vector<double>{2.0, 3.0}, Shape({2}), true);
    Tensor multiply_right(std::vector<double>{4.0, 5.0}, Shape({2}), true);
    multiply_left.multiply(multiply_right).sum().backward();
    assert(near(multiply_left.grad()[0], 4.0));
    assert(near(multiply_left.grad()[1], 5.0));
    assert(near(multiply_right.grad()[0], 2.0));
    assert(near(multiply_right.grad()[1], 3.0));

    Tensor scalar_multiply(std::vector<double>{2.0, 3.0}, Shape({2}), true);
    scalar_multiply.multiply(2.5).sum().backward();
    assert(near(scalar_multiply.grad()[0], 2.5));
    assert(near(scalar_multiply.grad()[1], 2.5));

    Tensor mean_input(std::vector<double>{2.0, 4.0}, Shape({2}), true);
    mean_input.mean().backward();
    assert(near(mean_input.grad()[0], 0.5));
    assert(near(mean_input.grad()[1], 0.5));

    Tensor tensor_pow_input(std::vector<double>{2.0, 3.0}, Shape({2}), true);
    tensor_pow_input.pow(3.0).sum().backward();
    assert(near(tensor_pow_input.grad()[0], 12.0));
    assert(near(tensor_pow_input.grad()[1], 27.0));

    Tensor matmul_left(std::vector<double>{1.0, 2.0, 3.0, 4.0}, Shape({2, 2}), true);
    Tensor matmul_right(std::vector<double>{5.0, 6.0}, Shape({2, 1}), true);
    matmul_left.matmul(matmul_right).sum().backward();
    assert(near(matmul_left.grad()[0], 5.0));
    assert(near(matmul_left.grad()[1], 6.0));
    assert(near(matmul_left.grad()[2], 5.0));
    assert(near(matmul_left.grad()[3], 6.0));
    assert(near(matmul_right.grad()[0], 4.0));
    assert(near(matmul_right.grad()[1], 6.0));

    Tensor row_left(std::vector<double>{1.0, 2.0, 3.0, 4.0}, Shape({2, 2}), true);
    Tensor row_right(std::vector<double>{10.0, 20.0}, Shape({1, 2}), true);
    row_left.addRowVector(row_right).sum().backward();
    assert(near(row_left.grad()[0], 1.0));
    assert(near(row_left.grad()[1], 1.0));
    assert(near(row_left.grad()[2], 1.0));
    assert(near(row_left.grad()[3], 1.0));
    assert(near(row_right.grad()[0], 2.0));
    assert(near(row_right.grad()[1], 2.0));

    Tensor relu_tensor(std::vector<double>{-1.0, 0.0, 2.0}, Shape({3}), true);
    relu_tensor.relu().sum().backward();
    assert(near(relu_tensor.grad()[0], 0.0));
    assert(near(relu_tensor.grad()[1], 0.0));
    assert(near(relu_tensor.grad()[2], 1.0));

    Tensor sigmoid_tensor(std::vector<double>{0.0}, Shape({1}), true);
    sigmoid_tensor.sigmoid().sum().backward();
    assert(near(sigmoid_tensor.grad()[0], 0.25));

    Tensor tanh_tensor(std::vector<double>{0.0}, Shape({1}), true);
    tanh_tensor.tanh().sum().backward();
    assert(near(tanh_tensor.grad()[0], 1.0));

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
