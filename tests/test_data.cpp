#include "neuroforge/data/CSVDataset.hpp"
#include "neuroforge/data/DataLoader.hpp"
#include "neuroforge/data/Dataset.hpp"
#include "neuroforge/data/TrainTestSplitter.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

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

std::filesystem::path writeFile(const std::string& name, const std::string& contents) {
    std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream file(path);
    file << contents;
    return path;
}

int main() {
    Tensor features = Tensor::fromVector({
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    });
    Tensor labels = Tensor::fromVector({
        {10.0},
        {20.0},
        {30.0}
    });
    Dataset dataset(features, labels);
    assert(dataset.size() == 3);
    assert(dataset.features().shape() == Shape({3, 2}));
    assert(dataset.labels().shape() == Shape({3, 1}));

    Batch sample = dataset.sample(1);
    assert(sample.features.shape() == Shape({1, 2}));
    assert(sample.labels.shape() == Shape({1, 1}));
    assert(sample.features.at(0, 0) == 3.0);
    assert(sample.features.at(0, 1) == 4.0);
    assert(sample.labels.at(0, 0) == 20.0);

    Dataset subset = dataset.subset({2, 0});
    assert(subset.size() == 2);
    assert(subset.features().at(0, 0) == 5.0);
    assert(subset.features().at(1, 0) == 1.0);
    assert(subset.labels().at(0, 0) == 30.0);

    expectThrows<std::invalid_argument>([] {
        Dataset invalid(Tensor::fromVector({1.0, 2.0}), Tensor::fromVector({1.0, 2.0}));
    });
    expectThrows<std::invalid_argument>([] {
        Dataset invalid(Tensor::fromVector({{1.0}, {2.0}}), Tensor::fromVector({{1.0}}));
    });
    expectThrows<std::out_of_range>([&] {
        dataset.sample(3);
    });
    expectThrows<std::invalid_argument>([&] {
        dataset.subset({});
    });
    expectThrows<std::out_of_range>([&] {
        dataset.subset({0, 5});
    });

    std::filesystem::path valid_path = writeFile("neuroforge_valid.csv", "hours,previous,result\n1,50,55\n2,55,60\n3,60,66\n");
    CSVDataset csv(valid_path.string(), 2);
    assert(csv.size() == 3);
    assert(csv.features().shape() == Shape({3, 2}));
    assert(csv.labels().shape() == Shape({3, 1}));
    assert(csv.features().at(0, 0) == 1.0);
    assert(csv.features().at(0, 1) == 50.0);
    assert(csv.labels().at(0, 0) == 55.0);

    std::filesystem::path label_middle_path = writeFile("neuroforge_label_middle.csv", "1,10,100\n2,20,200\n");
    CSVDataset label_middle(label_middle_path.string(), 1, false);
    assert(label_middle.features().shape() == Shape({2, 2}));
    assert(label_middle.labels().at(0, 0) == 10.0);
    assert(label_middle.features().at(0, 0) == 1.0);
    assert(label_middle.features().at(0, 1) == 100.0);

    std::filesystem::path invalid_numeric_path = writeFile("neuroforge_invalid_numeric.csv", "a,b,c\n1,x,3\n");
    expectThrows<std::invalid_argument>([&] {
        CSVDataset invalid(invalid_numeric_path.string(), 2);
    });

    std::filesystem::path inconsistent_path = writeFile("neuroforge_inconsistent.csv", "a,b,c\n1,2,3\n4,5\n");
    expectThrows<std::invalid_argument>([&] {
        CSVDataset invalid(inconsistent_path.string(), 2);
    });

    expectThrows<std::invalid_argument>([] {
        CSVDataset missing("missing_neuroforge_file.csv", 0);
    });
    expectThrows<std::invalid_argument>([&] {
        CSVDataset invalid(valid_path.string(), 3);
    });

    DataLoader loader(dataset, 2);
    std::vector<Batch> batches = loader.batches();
    assert(batches.size() == 2);
    assert(batches[0].features.shape() == Shape({2, 2}));
    assert(batches[0].labels.shape() == Shape({2, 1}));
    assert(batches[1].features.shape() == Shape({1, 2}));
    assert(batches[0].features.at(0, 0) == 1.0);
    assert(batches[0].features.at(1, 0) == 3.0);
    assert(batches[1].features.at(0, 0) == 5.0);

    DataLoader shuffled_a(dataset, 1, true, 42);
    DataLoader shuffled_b(dataset, 1, true, 42);
    std::vector<Batch> shuffled_a_batches = shuffled_a.batches();
    std::vector<Batch> shuffled_b_batches = shuffled_b.batches();
    assert(shuffled_a_batches.size() == shuffled_b_batches.size());
    for (size_t index = 0; index < shuffled_a_batches.size(); ++index) {
        assert(shuffled_a_batches[index].features.at(0, 0) == shuffled_b_batches[index].features.at(0, 0));
    }

    expectThrows<std::invalid_argument>([&] {
        DataLoader invalid(dataset, 0);
    });

    TrainTestSplit split = TrainTestSplitter::split(dataset, 0.67, false);
    assert(split.train.size() == 2);
    assert(split.test.size() == 1);
    assert(split.train.features().at(0, 0) == 1.0);
    assert(split.train.features().at(1, 0) == 3.0);
    assert(split.test.features().at(0, 0) == 5.0);

    TrainTestSplit shuffled_split_a = TrainTestSplitter::split(dataset, 0.67, true, 12);
    TrainTestSplit shuffled_split_b = TrainTestSplitter::split(dataset, 0.67, true, 12);
    assert(shuffled_split_a.train.features().at(0, 0) == shuffled_split_b.train.features().at(0, 0));
    assert(shuffled_split_a.test.features().at(0, 0) == shuffled_split_b.test.features().at(0, 0));

    expectThrows<std::invalid_argument>([&] {
        TrainTestSplitter::split(Dataset(Tensor::fromVector({{1.0}}), Tensor::fromVector({{1.0}})), 0.8);
    });
    expectThrows<std::invalid_argument>([&] {
        TrainTestSplitter::split(dataset, 0.0);
    });
    expectThrows<std::invalid_argument>([&] {
        TrainTestSplitter::split(dataset, 1.0);
    });

    return 0;
}
