#include "neuroforge/neuroforge.hpp"

#include <iostream>
#include <memory>

using namespace neuroforge;

int main() {
    Random::seed(42);

    CSVDataset dataset("data/population_regression.csv", 1);

    Sequential model;
    model.add(std::make_shared<Linear>(1, 1));

    MSELoss loss;
    SGD optimizer(model.parameters(), 0.5);
    Trainer trainer(model, loss, optimizer);

    const double initial_loss = trainer.evaluateLoss(dataset.features(), dataset.labels());

    TrainingConfig config;
    config.epochs = 3000;
    TrainingHistory history = trainer.fit(dataset.features(), dataset.labels(), config);

    Tensor input = Tensor::fromVector({{1.0}});
    Tensor prediction = model.forward(input);
    const double predicted_population = prediction.at(0, 0) * 1000000.0;

    std::cout << model.summary() << "\n";
    std::cout << "Initial loss: " << initial_loss << "\n";
    std::cout << "Final loss: " << history.finalLoss() << "\n";
    std::cout << "Predicted population for 2025: " << predicted_population << "\n";

    return history.finalLoss() < initial_loss
        && predicted_population > 150000.0
        && predicted_population < 160000.0 ? 0 : 1;
}
