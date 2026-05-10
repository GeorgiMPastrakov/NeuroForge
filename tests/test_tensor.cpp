#include "neuroforge/core/Tensor.hpp"

#include <cassert>
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

    return 0;
}
