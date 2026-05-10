#include "neuroforge/data/CSVDataset.hpp"
#include "neuroforge/data/Dataset.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
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

    return 0;
}
