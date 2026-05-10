#include "neuroforge/core/Shape.hpp"

#include <cassert>
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

int main() {
    Shape vector_shape({4});
    assert(vector_shape.rank() == 1);
    assert(vector_shape.size() == 4);
    assert(vector_shape[0] == 4);
    assert(vector_shape.toString() == "[4]");

    Shape matrix_shape({2, 3});
    assert(matrix_shape.rank() == 2);
    assert(matrix_shape.size() == 6);
    assert(matrix_shape.rows() == 2);
    assert(matrix_shape.cols() == 3);
    assert(matrix_shape[0] == 2);
    assert(matrix_shape[1] == 3);
    assert(matrix_shape.dimensions().size() == 2);
    assert(matrix_shape.toString() == "[2, 3]");

    assert(matrix_shape == Shape({2, 3}));
    assert(matrix_shape != Shape({3, 2}));

    expectThrows<std::invalid_argument>([] { Shape({}); });
    expectThrows<std::invalid_argument>([] { Shape({0}); });
    expectThrows<std::invalid_argument>([] { Shape({2, 0}); });
    expectThrows<std::invalid_argument>([] { Shape({2, 3, 4}); });
    expectThrows<std::invalid_argument>([] { Shape({4}).rows(); });
    expectThrows<std::invalid_argument>([] { Shape({4}).cols(); });
    expectThrows<std::out_of_range>([] { Shape({4})[1]; });

    return 0;
}
