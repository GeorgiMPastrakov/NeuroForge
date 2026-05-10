#pragma once

#include "neuroforge/core/Shape.hpp"

#include <cstddef>
#include <initializer_list>
#include <string>
#include <vector>

namespace neuroforge {

class Tensor {
public:
    Tensor(std::vector<double> data, Shape shape);

    static Tensor fromVector(std::initializer_list<std::initializer_list<double>> values);
    static Tensor fromVector(const std::vector<double>& values);
    static Tensor fromVector(const std::vector<std::vector<double>>& values);
    static Tensor zeros(const Shape& shape);
    static Tensor ones(const Shape& shape);
    static Tensor full(const Shape& shape, double value);

    const Shape& shape() const;
    size_t rank() const;
    size_t size() const;

    const std::vector<double>& data() const;
    std::vector<double>& data();

    double at(size_t index) const;
    double& at(size_t index);
    double at(size_t row, size_t col) const;
    double& at(size_t row, size_t col);

    Tensor add(const Tensor& other) const;
    Tensor subtract(const Tensor& other) const;
    Tensor multiply(const Tensor& other) const;
    Tensor multiply(double scalar) const;
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    Tensor sum() const;
    Tensor mean() const;
    double item() const;

private:
    std::vector<double> data_;
    Shape shape_;

    size_t flatIndex(size_t row, size_t col) const;
    void requireSameShape(const Tensor& other, const std::string& operation) const;
};

}
