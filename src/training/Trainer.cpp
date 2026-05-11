#include "neuroforge/training/Trainer.hpp"

#include <stdexcept>

namespace neuroforge {

Trainer::Trainer(Module& model, LossFunction& loss, Optimizer& optimizer)
    : model_(model), loss_(loss), optimizer_(optimizer) {
}

TrainingHistory Trainer::fit(const Tensor& input, const Tensor& target, const TrainingConfig& config) {
    if (config.epochs == 0) {
        throw std::invalid_argument("TrainingConfig epochs must be greater than zero.");
    }

    TrainingHistory history;
    model_.train();

    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        optimizer_.zeroGrad();
        Tensor prediction = model_.forward(input);
        Tensor loss_value = loss_.forward(prediction, target);
        Tensor grad = loss_.backward();
        model_.backward(grad);
        optimizer_.step();
        history.addLoss(epoch, loss_value.item());
    }

    return history;
}

TrainingHistory Trainer::fitAutograd(const Tensor& input, const Tensor& target, const TrainingConfig& config) {
    if (config.epochs == 0) {
        throw std::invalid_argument("TrainingConfig epochs must be greater than zero.");
    }

    TrainingHistory history;
    model_.train();

    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        optimizer_.zeroGrad();
        model_.enableAutograd();
        Tensor prediction = model_.forward(input);
        Tensor loss_value = loss_.forward(prediction, target);
        loss_value.backward();
        model_.syncAutogradGradients();
        optimizer_.step();
        history.addLoss(epoch, loss_value.item());
    }

    return history;
}

TrainingHistory Trainer::fit(const DataLoader& data_loader, const TrainingConfig& config) {
    if (config.epochs == 0) {
        throw std::invalid_argument("TrainingConfig epochs must be greater than zero.");
    }

    TrainingHistory history;
    model_.train();

    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        double weighted_loss = 0.0;
        size_t sample_count = 0;

        for (const Batch& batch : data_loader.batches()) {
            optimizer_.zeroGrad();
            Tensor prediction = model_.forward(batch.features);
            Tensor loss_value = loss_.forward(prediction, batch.labels);
            Tensor grad = loss_.backward();
            model_.backward(grad);
            optimizer_.step();
            const size_t rows = batch.features.shape().rows();
            weighted_loss += loss_value.item() * static_cast<double>(rows);
            sample_count += rows;
        }

        if (sample_count == 0) {
            throw std::invalid_argument("Trainer DataLoader fit requires at least one sample.");
        }

        history.addLoss(epoch, weighted_loss / static_cast<double>(sample_count));
    }

    return history;
}

TrainingHistory Trainer::fitAutograd(const DataLoader& data_loader, const TrainingConfig& config) {
    if (config.epochs == 0) {
        throw std::invalid_argument("TrainingConfig epochs must be greater than zero.");
    }

    TrainingHistory history;
    model_.train();

    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        double weighted_loss = 0.0;
        size_t sample_count = 0;

        for (const Batch& batch : data_loader.batches()) {
            optimizer_.zeroGrad();
            model_.enableAutograd();
            Tensor prediction = model_.forward(batch.features);
            Tensor loss_value = loss_.forward(prediction, batch.labels);
            loss_value.backward();
            model_.syncAutogradGradients();
            optimizer_.step();
            const size_t rows = batch.features.shape().rows();
            weighted_loss += loss_value.item() * static_cast<double>(rows);
            sample_count += rows;
        }

        if (sample_count == 0) {
            throw std::invalid_argument("Trainer DataLoader fitAutograd requires at least one sample.");
        }

        history.addLoss(epoch, weighted_loss / static_cast<double>(sample_count));
    }

    return history;
}

double Trainer::evaluateLoss(const Tensor& input, const Tensor& target) {
    const bool was_training = model_.isTraining();
    model_.eval();
    Tensor prediction = model_.forward(input);
    const double value = loss_.forward(prediction, target).item();

    if (was_training) {
        model_.train();
    } else {
        model_.eval();
    }

    return value;
}

double Trainer::evaluateLoss(const DataLoader& data_loader) {
    const bool was_training = model_.isTraining();
    model_.eval();
    double weighted_loss = 0.0;
    size_t sample_count = 0;

    for (const Batch& batch : data_loader.batches()) {
        Tensor prediction = model_.forward(batch.features);
        Tensor loss_value = loss_.forward(prediction, batch.labels);
        const size_t rows = batch.features.shape().rows();
        weighted_loss += loss_value.item() * static_cast<double>(rows);
        sample_count += rows;
    }

    if (was_training) {
        model_.train();
    } else {
        model_.eval();
    }

    if (sample_count == 0) {
        throw std::invalid_argument("Trainer DataLoader evaluateLoss requires at least one sample.");
    }

    return weighted_loss / static_cast<double>(sample_count);
}

}
