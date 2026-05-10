#include "neuroforge/neuroforge.hpp"

#include <iostream>
#include <memory>

using namespace neuroforge;

int main() {
    Random::seed(21);

    CSVDataset dataset("data/linear_regression.csv", 2);
    TrainTestSplit split = TrainTestSplitter::split(dataset, 0.75, false);

    Sequential model;
    model.add(std::make_shared<Linear>(2, 1));

    MSELoss loss;
    SGD optimizer(model.parameters(), 0.1);
    Trainer trainer(model, loss, optimizer);

    double initial_loss = trainer.evaluateLoss(split.train.features(), split.train.labels());

    TrainingConfig config;
    config.epochs = 1000;
    TrainingHistory history = trainer.fit(split.train.features(), split.train.labels(), config);

    Tensor train_prediction = model.forward(split.train.features());
    Tensor test_prediction = model.forward(split.test.features());
    double train_mse = mse(train_prediction, split.train.labels());
    double test_mse = mse(test_prediction, split.test.labels());

    ModelSaver::save(model, "build/neuroforge_linear_regression_model.txt");

    std::cout << model.summary() << "\n";
    std::cout << "Initial train loss: " << initial_loss << "\n";
    std::cout << "Final train loss: " << history.finalLoss() << "\n";
    std::cout << "Train MSE: " << train_mse << "\n";
    std::cout << "Test MSE: " << test_mse << "\n";
    std::cout << "Saved model: build/neuroforge_linear_regression_model.txt\n";

    return history.finalLoss() < initial_loss ? 0 : 1;
}
