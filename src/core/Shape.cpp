#include "neuroforge/core/Shape.hpp"

#include <numeric>
#include <sstream>
#include <stdexcept>

namespace neuroforge {

Shape::Shape(std::initializer_list<size_t> dimensions)
    : Shape(std::vector<size_t>(dimensions)) {
}

Shape::Shape(std::vector<size_t> dimensions)
    : dimensions_(std::move(dimensions)) {
    if (dimensions_.empty()) {
        throw std::invalid_argument("Shape must have rank 1 or 2.");
    }

    if (dimensions_.size() > 2) {
        throw std::invalid_argument("Shape rank greater than 2 is not supported.");
    }

    for (size_t dimension : dimensions_) {
        if (dimension == 0) {
            throw std::invalid_argument("Shape dimensions must be greater than zero.");
        }
    }
}

size_t Shape::rank() const {
    return dimensions_.size();
}

size_t Shape::size() const {
    return std::accumulate(dimensions_.begin(), dimensions_.end(), size_t{1}, std::multiplies<size_t>());
}

size_t Shape::rows() const {
    if (rank() != 2) {
        throw std::invalid_argument("Rows are only available for rank 2 shapes.");
    }

    return dimensions_[0];
}

size_t Shape::cols() const {
    if (rank() != 2) {
        throw std::invalid_argument("Columns are only available for rank 2 shapes.");
    }

    return dimensions_[1];
}

size_t Shape::operator[](size_t index) const {
    if (index >= dimensions_.size()) {
        throw std::out_of_range("Shape index out of range.");
    }

    return dimensions_[index];
}

const std::vector<size_t>& Shape::dimensions() const {
    return dimensions_;
}

std::string Shape::toString() const {
    std::ostringstream stream;
    stream << "[";

    for (size_t index = 0; index < dimensions_.size(); ++index) {
        if (index > 0) {
            stream << ", ";
        }

        stream << dimensions_[index];
    }

    stream << "]";
    return stream.str();
}

bool Shape::operator==(const Shape& other) const {
    return dimensions_ == other.dimensions_;
}

bool Shape::operator!=(const Shape& other) const {
    return !(*this == other);
}

}
