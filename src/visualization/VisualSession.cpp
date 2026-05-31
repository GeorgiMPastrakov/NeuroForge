#include "neuroforge/visualization/VisualSession.hpp"

#include "neuroforge/core/Random.hpp"
#include "neuroforge/data/CSVDataset.hpp"
#include "neuroforge/data/DataLoader.hpp"
#include "neuroforge/losses/BinaryCrossEntropyLoss.hpp"
#include "neuroforge/losses/CrossEntropyLoss.hpp"
#include "neuroforge/losses/LossFunction.hpp"
#include "neuroforge/losses/MSELoss.hpp"
#include "neuroforge/nn/Linear.hpp"
#include "neuroforge/nn/Sigmoid.hpp"
#include "neuroforge/optim/Adam.hpp"
#include "neuroforge/optim/Optimizer.hpp"
#include "neuroforge/optim/SGD.hpp"
#include "neuroforge/serialization/ModelLoader.hpp"
#include "neuroforge/training/Trainer.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace neuroforge {

VisualSession::VisualSession() {
    resetXorDemo();
}

bool VisualSession::resetXorDemo() {
    try {
        Random::seed(42);
        Sequential model;
        model.add(std::make_shared<Linear>(2, 4));
        model.add(std::make_shared<Sigmoid>());
        model.add(std::make_shared<Linear>(4, 1));
        model.add(std::make_shared<Sigmoid>());

        Dataset dataset(
            Tensor::fromVector({
                {0.0, 0.0},
                {0.0, 1.0},
                {1.0, 0.0},
                {1.0, 1.0}
            }),
            Tensor::fromVector({
                {0.0},
                {1.0},
                {1.0},
                {0.0}
            })
        );

        MSELoss loss;
        SGD optimizer(model.parameters(), 1.0);
        Trainer trainer(model, loss, optimizer);
        TrainingConfig config;
        config.epochs = 3000;
        config.verbose = false;

        history_ = trainer.fit(dataset.features(), dataset.labels(), config);
        model_ = std::move(model);
        dataset_ = std::move(dataset);
        status_ = "Loaded XOR demo.";
        refreshPredictions();
        return true;
    } catch (const std::exception& exception) {
        setError(exception);
        return false;
    }
}

bool VisualSession::loadModel(const std::string& path) {
    try {
        model_ = ModelLoader::load(path);
        status_ = "Loaded model: " + path;
        refreshPredictions();
        return true;
    } catch (const std::exception& exception) {
        setError(exception);
        return false;
    }
}

bool VisualSession::loadCsvDataset(const std::string& path, size_t label_column, bool has_header) {
    try {
        dataset_ = CSVDataset(path, label_column, has_header);
        status_ = "Loaded dataset: " + path;
        refreshPredictions();
        return true;
    } catch (const std::exception& exception) {
        setError(exception);
        return false;
    }
}

bool VisualSession::trainCurrent(const VisualTrainingConfig& config) {
    try {
        if (!hasModel()) {
            throw std::invalid_argument("VisualSession train requires a model.");
        }

        if (!dataset_.has_value()) {
            throw std::invalid_argument("VisualSession train requires a dataset.");
        }

        if (config.epochs == 0) {
            throw std::invalid_argument("VisualSession train epochs must be greater than zero.");
        }

        Dataset training_dataset = trainingDatasetFor(config);
        const size_t batch_size = config.batch_size == 0 ? training_dataset.size() : config.batch_size;
        DataLoader data_loader(training_dataset, batch_size, config.shuffle, config.seed);

        std::unique_ptr<LossFunction> loss;

        if (config.loss == VisualLossType::MSE) {
            loss = std::make_unique<MSELoss>();
        } else if (config.loss == VisualLossType::BinaryCrossEntropy) {
            loss = std::make_unique<BinaryCrossEntropyLoss>();
        } else {
            loss = std::make_unique<CrossEntropyLoss>();
        }

        std::unique_ptr<Optimizer> optimizer;

        if (config.optimizer == VisualOptimizerType::SGD) {
            optimizer = std::make_unique<SGD>(model_.parameters(), config.learning_rate);
        } else {
            optimizer = std::make_unique<Adam>(model_.parameters(), config.learning_rate);
        }

        TrainingConfig training_config;
        training_config.epochs = config.epochs;
        training_config.verbose = false;

        Trainer trainer(model_, *loss, *optimizer);
        history_ = trainer.fit(data_loader, training_config);
        status_ = "Training complete.";
        refreshPredictions();
        return true;
    } catch (const std::exception& exception) {
        setError(exception);
        return false;
    }
}

void VisualSession::refreshPredictions() {
    predictions_.reset();

    if (!dataset_.has_value() || !hasModel()) {
        return;
    }

    const bool was_training = model_.isTraining();
    model_.eval();
    predictions_ = model_.forward(dataset_->features());

    if (was_training) {
        model_.train();
    } else {
        model_.eval();
    }
}

Sequential& VisualSession::model() {
    return model_;
}

const Sequential& VisualSession::model() const {
    return model_;
}

const std::optional<Dataset>& VisualSession::dataset() const {
    return dataset_;
}

const TrainingHistory& VisualSession::history() const {
    return history_;
}

const std::optional<Tensor>& VisualSession::predictions() const {
    return predictions_;
}

const std::string& VisualSession::status() const {
    return status_;
}

bool VisualSession::hasModel() const {
    return model_.size() > 0;
}

bool VisualSession::hasDataset() const {
    return dataset_.has_value();
}

Dataset VisualSession::trainingDatasetFor(const VisualTrainingConfig& config) const {
    if (!dataset_.has_value()) {
        throw std::invalid_argument("VisualSession training dataset is missing.");
    }

    if (config.loss != VisualLossType::CrossEntropy) {
        return dataset_.value();
    }

    return Dataset(dataset_->features(), oneHotLabels(config.class_count));
}

Tensor VisualSession::oneHotLabels(size_t class_count) const {
    if (!dataset_.has_value()) {
        throw std::invalid_argument("VisualSession labels require a dataset.");
    }

    if (class_count == 0) {
        throw std::invalid_argument("VisualSession class count must be greater than zero.");
    }

    const Tensor& labels = dataset_->labels();

    if (labels.shape().cols() == class_count) {
        return labels;
    }

    if (labels.shape().cols() != 1) {
        throw std::invalid_argument("VisualSession CrossEntropy labels must be scalar class indices or one-hot rows.");
    }

    std::vector<double> values(labels.shape().rows() * class_count, 0.0);

    for (size_t row = 0; row < labels.shape().rows(); ++row) {
        const double value = labels.at(row, 0);
        const auto class_index = static_cast<size_t>(std::llround(value));

        if (std::fabs(value - static_cast<double>(class_index)) > 1e-9 || class_index >= class_count) {
            throw std::invalid_argument("VisualSession CrossEntropy scalar labels must be integer class indices.");
        }

        values[row * class_count + class_index] = 1.0;
    }

    return Tensor(std::move(values), Shape({labels.shape().rows(), class_count}));
}

void VisualSession::setError(const std::exception& exception) {
    status_ = "Error: ";
    status_ += exception.what();
}

}
