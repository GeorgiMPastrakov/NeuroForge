#include "neuroforge/data/CSVDataset.hpp"

#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace neuroforge {

namespace {

std::string trim(const std::string& value) {
    const size_t first = value.find_first_not_of(" \t\r\n");

    if (first == std::string::npos) {
        return "";
    }

    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

double parseNumber(const std::string& value, size_t line_number) {
    const std::string trimmed = trim(value);

    if (trimmed.empty()) {
        throw std::invalid_argument("CSV empty numeric value on line " + std::to_string(line_number) + ".");
    }

    char* end = nullptr;
    errno = 0;
    const double result = std::strtod(trimmed.c_str(), &end);

    if (errno != 0 || end == trimmed.c_str() || *end != '\0') {
        throw std::invalid_argument("CSV invalid numeric value on line " + std::to_string(line_number) + ".");
    }

    return result;
}

std::vector<std::string> splitLine(const std::string& line) {
    std::vector<std::string> values;
    std::string value;
    std::stringstream stream(line);

    while (std::getline(stream, value, ',')) {
        values.push_back(value);
    }

    if (!line.empty() && line.back() == ',') {
        values.emplace_back();
    }

    return values;
}

}

CSVDataset::CSVDataset(const std::string& path, size_t label_column, bool has_header)
    : Dataset(load(path, label_column, has_header)) {
}

Dataset CSVDataset::load(const std::string& path, size_t label_column, bool has_header) {
    std::ifstream file(path);

    if (!file) {
        throw std::invalid_argument("CSV file could not be opened: " + path);
    }

    std::vector<std::vector<double>> rows;
    std::string line;
    size_t line_number = 0;
    size_t expected_columns = 0;

    while (std::getline(file, line)) {
        ++line_number;

        if (has_header && line_number == 1) {
            continue;
        }

        if (trim(line).empty()) {
            continue;
        }

        std::vector<std::string> raw_values = splitLine(line);

        if (expected_columns == 0) {
            expected_columns = raw_values.size();

            if (expected_columns < 2) {
                throw std::invalid_argument("CSV dataset requires at least two columns.");
            }

            if (label_column >= expected_columns) {
                throw std::invalid_argument("CSV label column is out of range.");
            }
        }

        if (raw_values.size() != expected_columns) {
            throw std::invalid_argument("CSV row has inconsistent column count on line " + std::to_string(line_number) + ".");
        }

        std::vector<double> parsed;
        parsed.reserve(raw_values.size());

        for (const std::string& value : raw_values) {
            parsed.push_back(parseNumber(value, line_number));
        }

        rows.push_back(std::move(parsed));
    }

    if (rows.empty()) {
        throw std::invalid_argument("CSV dataset contains no rows.");
    }

    std::vector<double> feature_values;
    std::vector<double> label_values;
    feature_values.reserve(rows.size() * (expected_columns - 1));
    label_values.reserve(rows.size());

    for (const auto& row : rows) {
        for (size_t col = 0; col < row.size(); ++col) {
            if (col == label_column) {
                label_values.push_back(row[col]);
            } else {
                feature_values.push_back(row[col]);
            }
        }
    }

    return Dataset(
        Tensor(std::move(feature_values), Shape({rows.size(), expected_columns - 1})),
        Tensor(std::move(label_values), Shape({rows.size(), 1}))
    );
}

}
