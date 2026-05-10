#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

namespace neuroforge {

class Shape {
public:
    Shape(std::initializer_list<size_t> dimensions);
    explicit Shape(std::vector<size_t> dimensions);

    size_t rank() const;
    size_t size() const;
    size_t rows() const;
    size_t cols() const;
    size_t operator[](size_t index) const;
    const std::vector<size_t>& dimensions() const;
    std::string toString() const;

    bool operator==(const Shape& other) const;
    bool operator!=(const Shape& other) const;

private:
    std::vector<size_t> dimensions_;
};

}
