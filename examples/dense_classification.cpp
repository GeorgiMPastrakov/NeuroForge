#include "neuroforge/neuroforge.hpp"

#include <iostream>
#include <memory>

using namespace neuroforge;

namespace {

size_t argmaxRow(const Tensor& tensor, size_t row) {
    size_t best = 0;
    double best_value = tensor.at(row, 0);

    for (size_t col = 1; col < tensor.shape().cols(); ++col) {
        if (tensor.at(row, col) > best_value) {
            best = col;
            best_value = tensor.at(row, col);
        }
    }

    return best;
}

}

int main() {
    Random::seed(123);

    Tensor features = Tensor::fromVector({
        {-2.0, -1.8},
        {-1.8, -2.2},
        {-2.2, -1.6},
        {2.0, -2.0},
        {1.7, -2.4},
        {2.3, -1.7},
        {0.0, 2.0},
        {-0.3, 2.4},
        {0.4, 1.7}
    });

    Tensor labels = Tensor::fromVector({
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, 1.0}
    });

    Sequential model;
    model.add(std::make_shared<Linear>(2, 6));
    model.add(std::make_shared<LeakyReLU>(0.1));
    model.add(std::make_shared<Linear>(6, 3));

    CrossEntropyLoss loss;
    Adam optimizer(model.parameters(), 0.05);
    Trainer trainer(model, loss, optimizer);
    Dataset dataset(features, labels);
    DataLoader loader(dataset, 3, true, 17);

    const double initial_loss = trainer.evaluateLoss(loader);

    TrainingConfig config;
    config.epochs = 400;
    TrainingHistory history = trainer.fit(loader, config);

    Tensor probabilities = model.forward(features).softmaxRows();
    size_t correct = 0;

    for (size_t row = 0; row < features.shape().rows(); ++row) {
        const size_t predicted = argmaxRow(probabilities, row);
        const size_t expected = argmaxRow(labels, row);

        if (predicted == expected) {
            ++correct;
        }
    }

    std::cout << model.summary() << "\n";
    std::cout << "Initial loss: " << initial_loss << "\n";
    std::cout << "Final loss: " << history.finalLoss() << "\n";
    std::cout << "Accuracy: " << static_cast<double>(correct) / static_cast<double>(features.shape().rows()) << "\n";

    for (size_t row = 0; row < features.shape().rows(); ++row) {
        std::cout << "Input: [" << features.at(row, 0) << ", " << features.at(row, 1) << "] ";
        std::cout << "Prediction: " << argmaxRow(probabilities, row) << " ";
        std::cout << "Target: " << argmaxRow(labels, row) << "\n";
    }

    return correct == features.shape().rows() ? 0 : 1;
}
