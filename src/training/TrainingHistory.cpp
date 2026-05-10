#include "neuroforge/training/TrainingHistory.hpp"

#include <stdexcept>

namespace neuroforge {

void TrainingHistory::addLoss(size_t epoch, double loss) {
    epochs_.push_back(epoch);
    losses_.push_back(loss);
}

const std::vector<size_t>& TrainingHistory::epochs() const {
    return epochs_;
}

const std::vector<double>& TrainingHistory::losses() const {
    return losses_;
}

double TrainingHistory::finalLoss() const {
    if (losses_.empty()) {
        throw std::invalid_argument("TrainingHistory final loss is unavailable for empty history.");
    }

    return losses_.back();
}

bool TrainingHistory::empty() const {
    return losses_.empty();
}

size_t TrainingHistory::size() const {
    return losses_.size();
}

}
