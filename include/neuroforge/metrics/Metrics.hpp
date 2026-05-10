#pragma once

#include "neuroforge/core/Tensor.hpp"

namespace neuroforge {

double mse(const Tensor& prediction, const Tensor& target);
double mae(const Tensor& prediction, const Tensor& target);
double accuracy(const Tensor& prediction, const Tensor& target);
double binaryAccuracy(const Tensor& prediction, const Tensor& target);

}
