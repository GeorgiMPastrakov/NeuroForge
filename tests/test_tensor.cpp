#include "neuroforge/core/Tensor.hpp"
#include "neuroforge/core/Random.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

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

template <typename Exception, typename Function>
std::string thrownMessage(Function function) {
    try {
        function();
    } catch (const Exception& exception) {
        return exception.what();
    }

    assert(false);
    return "";
}

bool near(double left, double right) {
    return std::fabs(left - right) < 1e-9;
}

int main() {
    Tensor vector = Tensor::fromVector({1.0, 2.0, 3.0});
    assert(vector.rank() == 1);
    assert(vector.size() == 3);
    assert(vector.shape() == Shape({3}));
    assert(vector.at(0) == 1.0);
    assert(vector.at(2) == 3.0);
    vector.at(1) = 5.0;
    assert(vector.at(1) == 5.0);

    Tensor matrix = Tensor::fromVector({
        {1.0, 2.0},
        {3.0, 4.0}
    });
    assert(matrix.rank() == 2);
    assert(matrix.size() == 4);
    assert(matrix.shape() == Shape({2, 2}));
    assert(matrix.at(0, 0) == 1.0);
    assert(matrix.at(0, 1) == 2.0);
    assert(matrix.at(1, 0) == 3.0);
    assert(matrix.at(1, 1) == 4.0);
    matrix.at(1, 1) = 9.0;
    assert(matrix.at(1, 1) == 9.0);

    Tensor zeros = Tensor::zeros(Shape({2, 2}));
    assert(zeros.at(0, 0) == 0.0);
    assert(zeros.at(1, 1) == 0.0);

    Tensor ones = Tensor::ones(Shape({3}));
    assert(ones.at(0) == 1.0);
    assert(ones.at(2) == 1.0);

    Tensor full = Tensor::full(Shape({2, 2}), 7.0);
    assert(full.at(0, 0) == 7.0);
    assert(full.at(1, 1) == 7.0);
    assert(full.data().size() == 4);

    expectThrows<std::invalid_argument>([] { Tensor({1.0, 2.0}, Shape({3})); });
    expectThrows<std::invalid_argument>([] { Tensor::fromVector(std::vector<std::vector<double>>{}); });
    expectThrows<std::invalid_argument>([] { Tensor::fromVector(std::vector<std::vector<double>>{{}}); });
    expectThrows<std::invalid_argument>([] { Tensor::fromVector({{1.0}, {2.0, 3.0}}); });
    expectThrows<std::invalid_argument>([] { Tensor::fromVector({1.0, 2.0}).at(0, 0); });
    expectThrows<std::invalid_argument>([] { Tensor::fromVector(std::vector<std::vector<double>>{{1.0}}).at(0); });
    expectThrows<std::out_of_range>([] { Tensor::fromVector({1.0}).at(1); });
    expectThrows<std::out_of_range>([] { Tensor::fromVector(std::vector<std::vector<double>>{{1.0}}).at(1, 0); });

    Tensor left = Tensor::fromVector({
        {1.0, 2.0},
        {3.0, 4.0}
    });
    Tensor right = Tensor::fromVector({
        {5.0, 6.0},
        {7.0, 8.0}
    });

    Tensor added = left.add(right);
    assert(added.shape() == Shape({2, 2}));
    assert(added.at(0, 0) == 6.0);
    assert(added.at(1, 1) == 12.0);

    Tensor subtracted = right.subtract(left);
    assert(subtracted.at(0, 0) == 4.0);
    assert(subtracted.at(1, 1) == 4.0);

    Tensor multiplied = left.multiply(right);
    assert(multiplied.at(0, 0) == 5.0);
    assert(multiplied.at(1, 1) == 32.0);

    Tensor scaled = left.multiply(2.0);
    assert(scaled.at(0, 0) == 2.0);
    assert(scaled.at(1, 1) == 8.0);

    Tensor transposed = Tensor::fromVector({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    }).transpose();
    assert(transposed.shape() == Shape({3, 2}));
    assert(transposed.at(0, 0) == 1.0);
    assert(transposed.at(0, 1) == 4.0);
    assert(transposed.at(2, 0) == 3.0);
    assert(transposed.at(2, 1) == 6.0);

    assert(left.sum().shape() == Shape({1}));
    assert(left.sum().item() == 10.0);
    assert(near(left.mean().item(), 2.5));

    Tensor matmul_2x2 = left.matmul(right);
    assert(matmul_2x2.shape() == Shape({2, 2}));
    assert(matmul_2x2.at(0, 0) == 19.0);
    assert(matmul_2x2.at(0, 1) == 22.0);
    assert(matmul_2x2.at(1, 0) == 43.0);
    assert(matmul_2x2.at(1, 1) == 50.0);

    Tensor matmul_2x3_3x1 = Tensor::fromVector({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    }).matmul(Tensor::fromVector({
        {7.0},
        {8.0},
        {9.0}
    }));
    assert(matmul_2x3_3x1.shape() == Shape({2, 1}));
    assert(matmul_2x3_3x1.at(0, 0) == 50.0);
    assert(matmul_2x3_3x1.at(1, 0) == 122.0);

    std::string matmul_error = thrownMessage<std::invalid_argument>([] {
        Tensor::fromVector({
            {1.0, 2.0}
        }).matmul(Tensor::fromVector({
            {1.0},
            {2.0},
            {3.0}
        }));
    });
    assert(matmul_error.find("[1, 2]") != std::string::npos);
    assert(matmul_error.find("[3, 1]") != std::string::npos);

    expectThrows<std::invalid_argument>([] { Tensor::fromVector({1.0}).add(Tensor::fromVector({1.0, 2.0})); });
    expectThrows<std::invalid_argument>([] { Tensor::fromVector({1.0}).subtract(Tensor::fromVector({1.0, 2.0})); });
    expectThrows<std::invalid_argument>([] { Tensor::fromVector({1.0}).multiply(Tensor::fromVector({1.0, 2.0})); });
    expectThrows<std::invalid_argument>([] { Tensor::fromVector({1.0}).matmul(Tensor::fromVector({1.0})); });
    expectThrows<std::invalid_argument>([] { Tensor::fromVector({1.0}).transpose(); });
    expectThrows<std::invalid_argument>([] { Tensor::fromVector({1.0, 2.0}).item(); });

    Tensor activation_input = Tensor::fromVector({-1.0, 0.0, 1.0});
    Tensor relu = activation_input.relu();
    assert(relu.at(0) == 0.0);
    assert(relu.at(1) == 0.0);
    assert(relu.at(2) == 1.0);

    Tensor sigmoid = activation_input.sigmoid();
    assert(near(sigmoid.at(0), 1.0 / (1.0 + std::exp(1.0))));
    assert(near(sigmoid.at(1), 0.5));
    assert(near(sigmoid.at(2), 1.0 / (1.0 + std::exp(-1.0))));

    Tensor tanh = activation_input.tanh();
    assert(near(tanh.at(0), std::tanh(-1.0)));
    assert(near(tanh.at(1), 0.0));
    assert(near(tanh.at(2), std::tanh(1.0)));

    Random::seed(42);
    Tensor random_a = Random::uniform(Shape({2, 2}), -1.0, 1.0);
    Random::seed(42);
    Tensor random_b = Random::uniform(Shape({2, 2}), -1.0, 1.0);
    assert(random_a.shape() == Shape({2, 2}));
    assert(random_a.data() == random_b.data());
    expectThrows<std::invalid_argument>([] { Random::uniform(Shape({1}), 1.0, -1.0); });

    return 0;
}
