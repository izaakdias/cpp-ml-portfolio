#pragma once
#include <Eigen/Dense>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace portfolio {
using Json = nlohmann::json;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
struct Dataset {
  Matrix x;
  Vector y;
  std::vector<std::string> features;
};
struct Split {
  Dataset train, validation, test;
};
std::vector<std::vector<std::string>> read_csv(const std::filesystem::path& path);
Dataset read_dataset(const std::filesystem::path& path);
double number(const std::string& value);
void validate_matrix(const Matrix& values, bool allow_empty = false);
Split split_dataset(const Dataset& data, unsigned seed = 41, bool stratify = false);
Json vector_json(const Vector& vector);
Vector json_vector(const Json& json);
struct Standardizer {
  Vector mean, scale;
  static Standardizer fit(const Matrix& values);
  Matrix transform(const Matrix& values) const;
  Json json() const;
  static Standardizer from_json(const Json& json);
};
}  // namespace portfolio
