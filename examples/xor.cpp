#include "neuroforge/neuroforge.hpp"

#include <iostream>
#include <memory>

using namespace neuroforge;

int main() {
    Random::seed(42);

    Tensor input = Tensor::fromVector({
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    });
    Tensor target = Tensor::fromVector({
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    });

    Sequential model;
    model.add(std::make_shared<Linear>(2, 4));
    model.add(std::make_shared<Sigmoid>());
    model.add(std::make_shared<Linear>(4, 1));
    model.add(std::make_shared<Sigmoid>());

    MSELoss loss;
    SGD optimizer(model.parameters(), 1.0);
    Trainer trainer(model, loss, optimizer);

    const double initial_loss = trainer.evaluateLoss(input, target);

    TrainingConfig config;
    config.epochs = 3000;
    TrainingHistory history = trainer.fit(input, target, config);

    Tensor prediction = model.forward(input);

    std::cout << model.summary() << "\n";
    std::cout << "Initial loss: " << initial_loss << "\n";
    std::cout << "Final loss: " << history.finalLoss() << "\n";

    for (size_t row = 0; row < input.shape().rows(); ++row) {
        std::cout << "Input: [" << input.at(row, 0) << ", " << input.at(row, 1) << "] ";
        std::cout << "Prediction: " << prediction.at(row, 0) << " ";
        std::cout << "Target: " << target.at(row, 0) << "\n";
    }

    return history.finalLoss() < initial_loss ? 0 : 1;
}
