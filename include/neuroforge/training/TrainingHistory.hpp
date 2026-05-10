#pragma once

#include <cstddef>
#include <vector>

namespace neuroforge {

class TrainingHistory {
public:
    void addLoss(size_t epoch, double loss);
    const std::vector<size_t>& epochs() const;
    const std::vector<double>& losses() const;
    double finalLoss() const;
    bool empty() const;
    size_t size() const;

private:
    std::vector<size_t> epochs_;
    std::vector<double> losses_;
};

}
