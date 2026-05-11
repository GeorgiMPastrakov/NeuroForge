#include "neuroforge/visualization/VisualizationData.hpp"
#include "neuroforge/visualization/VisualSession.hpp"
#include "neuroforge/nn/Linear.hpp"
#include "neuroforge/nn/ReLU.hpp"
#include "neuroforge/nn/Sequential.hpp"
#include "neuroforge/nn/Softmax.hpp"
#include "neuroforge/serialization/ModelSaver.hpp"
#include "neuroforge/training/TrainingHistory.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

using namespace neuroforge;

template <typename Exception, typename Function>
void expectThrows(Function function) {
    bool thrown = false;

    try {
        function();
    } catch (const Exception&) {
        thrown = true;
    }

    assert(thrown);
}

bool near(double left, double right) {
    return std::fabs(left - right) < 1e-9;
}

int main() {
    Sequential empty;
    ModelSnapshot empty_snapshot = buildModelSnapshot(empty);
    assert(empty_snapshot.layers.empty());
    assert(empty_snapshot.total_parameters == 0);

    Sequential model;
    auto linear = std::make_shared<Linear>(2, 1);
    linear->weights().data().at(0, 0) = 1.0;
    linear->weights().data().at(1, 0) = 2.0;
    linear->bias().data().at(0, 0) = 0.5;
    linear->weights().grad().at(0, 0) = 3.0;
    linear->weights().grad().at(1, 0) = 4.0;
    model.add(linear);
    model.add(std::make_shared<ReLU>());

    ModelSnapshot model_snapshot = buildModelSnapshot(model);
    assert(model_snapshot.layers.size() == 2);
    assert(model_snapshot.layers[0].name == "Linear(2 -> 1)");
    assert(model_snapshot.layers[0].parameter_count == 3);
    assert(model_snapshot.layers[1].name == "ReLU()");
    assert(model_snapshot.total_parameters == 3);

    TrainingHistory history;
    LossHistorySnapshot empty_history = buildLossHistorySnapshot(history);
    assert(empty_history.epochs.empty());
    assert(empty_history.losses.empty());
    history.addLoss(1, 0.5);
    history.addLoss(2, 0.25);
    LossHistorySnapshot history_snapshot = buildLossHistorySnapshot(history);
    assert(history_snapshot.epochs.size() == 2);
    assert(history_snapshot.losses[1] == 0.25);

    Tensor tensor = Tensor::fromVector({1.0, 2.0, 3.0});
    TensorSnapshot tensor_snapshot = buildTensorSnapshot("tensor", tensor, 2);
    assert(tensor_snapshot.name == "tensor");
    assert(tensor_snapshot.shape == Shape({3}));
    assert(tensor_snapshot.values.size() == 2);
    assert(tensor_snapshot.original_size == 3);
    assert(tensor_snapshot.truncated);

    std::vector<GradientSnapshot> gradients = buildGradientSnapshots(model, 10);
    assert(gradients.size() == 2);
    assert(gradients[0].name == "weights");
    assert(near(gradients[0].norm, 5.0));
    assert(gradients[1].name == "bias");
    assert(near(gradients[1].norm, 0.0));

    DecisionBoundaryGrid grid = buildDecisionBoundaryGrid(model, 0.0, 1.0, 0.0, 1.0, 2, 3);
    assert(grid.rows == 2);
    assert(grid.cols == 3);
    assert(grid.supported);
    assert(grid.x_values.size() == 3);
    assert(grid.y_values.size() == 2);
    assert(grid.predictions.size() == 6);
    assert(grid.classes.size() == 6);
    assert(near(grid.predictions.front(), 0.5));

    Dataset binary_dataset(
        Tensor::fromVector({
            {0.0, 0.0},
            {1.0, 1.0}
        }),
        Tensor::fromVector({
            {0.0},
            {1.0}
        })
    );
    DatasetSnapshot binary_snapshot = buildDatasetSnapshot(binary_dataset);
    assert(binary_snapshot.supported);
    assert(binary_snapshot.points.size() == 2);
    assert(binary_snapshot.class_count == 2);

    expectThrows<std::invalid_argument>([] {
        Tensor tensor = Tensor::fromVector({1.0});
        buildTensorSnapshot("bad", tensor, 0);
    });
    expectThrows<std::invalid_argument>([&] {
        buildDecisionBoundaryGrid(model, 0.0, 1.0, 0.0, 1.0, 1, 2);
    });

    Tensor binary_prediction = Tensor::fromVector({
        {0.1},
        {0.9}
    });
    PredictionSnapshot binary_predictions = buildPredictionSnapshot(&binary_dataset, &binary_prediction, 2);
    assert(binary_predictions.supported);
    assert(binary_predictions.kind == PredictionKind::Binary);
    assert(binary_predictions.rows.size() == 2);
    assert(binary_predictions.rows[1].predicted_class == 1);

    Dataset multiclass_dataset(
        Tensor::fromVector({
            {0.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0}
        }),
        Tensor::fromVector({
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
        })
    );
    DatasetSnapshot multiclass_snapshot = buildDatasetSnapshot(multiclass_dataset);
    assert(multiclass_snapshot.supported);
    assert(multiclass_snapshot.class_count == 3);

    Tensor multiclass_prediction = Tensor::fromVector({
        {4.0, 1.0, 0.0},
        {0.0, 3.0, 1.0},
        {0.0, 1.0, 2.0}
    });
    PredictionSnapshot multiclass_predictions = buildPredictionSnapshot(&multiclass_dataset, &multiclass_prediction, 3);
    assert(multiclass_predictions.kind == PredictionKind::Multiclass);
    assert(multiclass_predictions.rows[2].predicted_class == 2);

    Sequential multiclass_model;
    multiclass_model.add(std::make_shared<Linear>(2, 3));
    multiclass_model.add(std::make_shared<Softmax>());
    DecisionBoundaryGrid multiclass_grid = buildDecisionBoundaryGrid(multiclass_model, multiclass_dataset, -1.0, 1.0, -1.0, 1.0, 3, 3);
    assert(multiclass_grid.supported);
    assert(multiclass_grid.classes.size() == 9);

    Dataset unsupported_dataset(
        Tensor(std::vector<double>{0.0, 1.0, 2.0}, Shape({1, 3})),
        Tensor::fromVector({
            {0.0}
        })
    );
    DatasetSnapshot unsupported_snapshot = buildDatasetSnapshot(unsupported_dataset);
    assert(!unsupported_snapshot.supported);
    DecisionBoundaryGrid unsupported_grid = buildDecisionBoundaryGrid(model, unsupported_dataset, 0.0, 1.0, 0.0, 1.0, 2, 2);
    assert(!unsupported_grid.supported);

    NetworkGraphSnapshot empty_graph = buildNetworkGraphSnapshot(empty, 12);
    assert(!empty_graph.supported);
    NetworkGraphSnapshot graph = buildNetworkGraphSnapshot(model, 12);
    assert(graph.supported);
    assert(graph.nodes.size() == 3);
    assert(graph.edges.size() == 2);
    Sequential large_model;
    large_model.add(std::make_shared<Linear>(16, 20));
    NetworkGraphSnapshot large_graph = buildNetworkGraphSnapshot(large_model, 12);
    assert(large_graph.supported);
    assert(large_graph.nodes.size() == 2);
    assert(large_graph.nodes[0].aggregate);
    assert(large_graph.nodes[1].aggregate);

    VisualSession session;
    assert(session.hasModel());
    assert(session.hasDataset());
    assert(session.predictions().has_value());
    assert(session.predictions()->shape() == Shape({4, 1}));
    assert(!session.status().empty());
    assert(!session.loadModel("missing_visual_session_model.txt"));
    assert(session.status().find("Error:") == 0);

    Sequential saved_model;
    saved_model.add(std::make_shared<Linear>(2, 2));
    std::filesystem::path model_path = std::filesystem::temp_directory_path() / "neuroforge_visual_session_model.txt";
    ModelSaver::save(saved_model, model_path.string());
    assert(session.loadModel(model_path.string()));
    assert(session.hasModel());

    std::filesystem::path csv_path = std::filesystem::temp_directory_path() / "neuroforge_visual_session.csv";
    {
        std::ofstream file(csv_path);
        file << "x0,x1,label\n";
        file << "0,0,0\n";
        file << "1,0,1\n";
        file << "0,1,1\n";
        file << "1,1,0\n";
    }
    assert(session.loadCsvDataset(csv_path.string(), 2, true));
    assert(session.hasDataset());
    assert(!session.loadCsvDataset(csv_path.string(), 9, true));
    assert(session.loadCsvDataset(csv_path.string(), 2, true));

    VisualTrainingConfig train_config;
    train_config.epochs = 2;
    train_config.learning_rate = 0.01;
    train_config.loss = VisualLossType::CrossEntropy;
    train_config.optimizer = VisualOptimizerType::Adam;
    train_config.class_count = 2;
    assert(session.trainCurrent(train_config));
    assert(!session.history().losses().empty());
    assert(session.predictions().has_value());

    return 0;
}
