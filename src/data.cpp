#include "portfolio/data.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>

namespace portfolio {
std::vector<std::vector<std::string>> read_csv(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("Cannot open CSV: " + path.string());
  std::vector<std::vector<std::string>> records;
  std::vector<std::string> row;
  std::string field;
  bool quoted = false, closed = false;
  char character;
  while (input.get(character)) {
    if (quoted) {
      if (character == '"') {
        if (input.peek() == '"') {
          input.get();
          field += '"';
        } else {
          quoted = false;
          closed = true;
        }
      } else
        field += character;
    } else if (character == '"') {
      if (!field.empty() || closed)
        throw std::invalid_argument("Unexpected CSV quote");
      quoted = true;
    } else if (character == ',' || character == '\n' || character == '\r') {
      row.push_back(field);
      field.clear();
      closed = false;
      if (character != ',') {
        if (character == '\r' && input.peek() == '\n')
          input.get();
        records.push_back(row);
        row.clear();
      }
    } else {
      if (closed)
        throw std::invalid_argument("Content after closing CSV quote");
      field += character;
    }
  }
  if (quoted)
    throw std::invalid_argument("Unclosed CSV quote");
  if (!field.empty() || !row.empty() || closed) {
    row.push_back(field);
    records.push_back(row);
  }
  if (records.empty())
    throw std::invalid_argument("Empty CSV");
  for (const auto& record : records)
    if (record.size() != records.front().size())
      throw std::invalid_argument("Inconsistent CSV column count");
  return records;
}
double number(const std::string& value) {
  std::size_t used = 0;
  double result = std::stod(value, &used);
  if (used != value.size() || !std::isfinite(result))
    throw std::invalid_argument("Invalid finite number: " + value);
  return result;
}
void validate_matrix(const Matrix& values, bool allow_empty) {
  if ((!allow_empty && values.rows() == 0) || values.cols() == 0 || !values.allFinite())
    throw std::invalid_argument("Expected a nonempty finite matrix");
}
Dataset read_dataset(const std::filesystem::path& path) {
  auto rows = read_csv(path);
  if (rows.size() < 2 || rows.front().size() < 2)
    throw std::invalid_argument("CSV requires feature columns and a final target column");
  std::set<std::string> names(rows.front().begin(), rows.front().end());
  if (names.size() != rows.front().size() || names.contains(""))
    throw std::invalid_argument("CSV headers must be unique and nonempty");
  Dataset data{Matrix(rows.size() - 1, rows.front().size() - 1),
               Vector(rows.size() - 1),
               {rows.front().begin(), rows.front().end() - 1}};
  for (std::size_t i = 1; i < rows.size(); ++i) {
    for (Eigen::Index j = 0; j < data.x.cols(); ++j) data.x(i - 1, j) = number(rows[i][j]);
    data.y(i - 1) = number(rows[i].back());
  }
  return data;
}
static Dataset subset(const Dataset& data, const std::vector<Eigen::Index>& indices) {
  Dataset result{Matrix(indices.size(), data.x.cols()), Vector(indices.size()), data.features};
  for (std::size_t i = 0; i < indices.size(); ++i) {
    result.x.row(i) = data.x.row(indices[i]);
    result.y(i) = data.y(indices[i]);
  }
  return result;
}
Split split_dataset(const Dataset& data, unsigned seed, bool stratify) {
  validate_matrix(data.x);
  if (data.x.rows() != data.y.size() || !data.y.allFinite() || data.x.rows() < 6)
    throw std::invalid_argument("At least six aligned examples required");
  std::map<double, std::vector<Eigen::Index>> groups;
  for (Eigen::Index i = 0; i < data.x.rows(); ++i) groups[stratify ? data.y(i) : 0.0].push_back(i);
  std::mt19937 rng(seed);
  std::vector<Eigen::Index> train, validation, test;
  for (auto& [label, indices] : groups) {
    (void)label;
    if (indices.size() < 3)
      throw std::invalid_argument("Each stratification group needs three examples");
    std::shuffle(indices.begin(), indices.end(), rng);
    auto a = std::max<std::size_t>(1, indices.size() * 6 / 10);
    auto b = std::max<std::size_t>(1, indices.size() * 2 / 10);
    if (a + b >= indices.size())
      a = indices.size() - b - 1;
    train.insert(train.end(), indices.begin(), indices.begin() + a);
    validation.insert(validation.end(), indices.begin() + a, indices.begin() + a + b);
    test.insert(test.end(), indices.begin() + a + b, indices.end());
  }
  return {subset(data, train), subset(data, validation), subset(data, test)};
}
Json vector_json(const Vector& vector) {
  return std::vector<double>(vector.data(), vector.data() + vector.size());
}
Vector json_vector(const Json& json) {
  auto values = json.get<std::vector<double>>();
  if (values.empty())
    throw std::invalid_argument("Empty serialized vector");
  Vector vector = Eigen::Map<Vector>(values.data(), values.size());
  if (!vector.allFinite())
    throw std::invalid_argument("Nonfinite serialized vector");
  return vector;
}
Standardizer Standardizer::fit(const Matrix& values) {
  validate_matrix(values);
  Standardizer result;
  result.mean = values.colwise().mean();
  result.scale = ((values.rowwise() - result.mean.transpose()).array().square().colwise().mean())
                     .sqrt()
                     .matrix();
  for (auto& value : result.scale)
    if (value < 1e-12)
      value = 1.0;
  return result;
}
Matrix Standardizer::transform(const Matrix& values) const {
  validate_matrix(values);
  if (values.cols() != mean.size() || scale.size() != mean.size() || (scale.array() <= 0).any())
    throw std::invalid_argument("Feature contract mismatch");
  return (values.rowwise() - mean.transpose()).array().rowwise() / scale.transpose().array();
}
Json Standardizer::json() const {
  return {{"mean", vector_json(mean)}, {"scale", vector_json(scale)}};
}
Standardizer Standardizer::from_json(const Json& json) {
  Standardizer result{json_vector(json.at("mean")), json_vector(json.at("scale"))};
  if (result.mean.size() != result.scale.size() || (result.scale.array() <= 0).any())
    throw std::invalid_argument("Invalid standardizer");
  return result;
}
}  // namespace portfolio
