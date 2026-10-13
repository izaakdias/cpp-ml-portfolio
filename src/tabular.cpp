#include "portfolio/tabular.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace portfolio {
static Matrix design(const Matrix& x) {
  validate_matrix(x);
  Matrix result(x.rows(), x.cols() + 1);
  result.col(0).setOnes();
  result.rightCols(x.cols()) = x;
  return result;
}
static void validate_targets(const Matrix& x, const Vector& y, bool binary = false) {
  validate_matrix(x);
  if (x.rows() != y.size() || !y.allFinite())
    throw std::invalid_argument("Aligned finite targets required");
  if (binary && ((y.array() != 0 && y.array() != 1).any() || y.sum() == 0 || y.sum() == y.size()))
    throw std::invalid_argument("Both binary classes required");
}
void LinearRegression::fit(const Matrix& x, const Vector& y, double ridge) {
  validate_targets(x, y);
  if (!std::isfinite(ridge) || ridge < 0)
    throw std::invalid_argument("Nonnegative ridge required");
  Matrix a = Matrix::Zero(x.rows() + x.cols(), x.cols() + 1);
  a.topRows(x.rows()) = design(x);
  a.bottomRightCorner(x.cols(), x.cols()).diagonal().setConstant(std::sqrt(ridge));
  Vector b = Vector::Zero(a.rows());
  b.head(y.size()) = y;
  weights = a.colPivHouseholderQr().solve(b);
  if (!weights.allFinite())
    throw std::runtime_error("Linear fit produced invalid weights");
}
Vector LinearRegression::predict(const Matrix& x) const {
  if (weights.size() != x.cols() + 1)
    throw std::invalid_argument("Model is unfitted or feature count differs");
  return design(x) * weights;
}
static double sigmoid(double z) {
  return z >= 0 ? 1 / (1 + std::exp(-z)) : std::exp(z) / (1 + std::exp(z));
}
void LogisticRegression::fit(const Matrix& x, const Vector& y, int epochs, double learning_rate,
                             double l2, bool balanced) {
  validate_targets(x, y, true);
  if (epochs < 1 || !std::isfinite(learning_rate) || learning_rate <= 0 || !std::isfinite(l2) ||
      l2 < 0)
    throw std::invalid_argument("Invalid optimizer parameters");
  Matrix a = design(x);
  weights = Vector::Zero(a.cols());
  Vector sample_weights = Vector::Ones(y.size());
  if (balanced)
    for (Eigen::Index i = 0; i < y.size(); ++i)
      sample_weights(i) = y.size() / (2 * (y(i) == 1 ? y.sum() : y.size() - y.sum()));
  for (int epoch = 0; epoch < epochs; ++epoch) {
    Vector probabilities = (a * weights).unaryExpr([](double z) { return sigmoid(z); });
    Vector gradient = a.transpose() *
                      ((probabilities - y).array() * sample_weights.array()).matrix() /
                      sample_weights.sum();
    gradient.tail(x.cols()) += l2 * weights.tail(x.cols());
    weights -= learning_rate * gradient;
    if (!weights.allFinite())
      throw std::runtime_error("Optimizer diverged");
  }
}
Vector LogisticRegression::predict(const Matrix& x) const {
  if (weights.size() != x.cols() + 1)
    throw std::invalid_argument("Model is unfitted or feature count differs");
  return (design(x) * weights).unaryExpr([](double z) { return sigmoid(z); });
}
Json LogisticRegression::json() const {
  return {{"kind", "logistic_regression"}, {"weights", vector_json(weights)}};
}
LogisticRegression LogisticRegression::from_json(const Json& json) {
  if (json.at("kind") != "logistic_regression")
    throw std::invalid_argument("Unexpected model kind");
  return {json_vector(json.at("weights"))};
}
void KNearestNeighbors::fit(const Matrix& x, const Vector& y, int k) {
  validate_targets(x, y, true);
  if (k < 1 || k > x.rows())
    throw std::invalid_argument("Invalid neighbor count");
  features = x;
  labels = y;
  neighbors = k;
}
Vector KNearestNeighbors::predict(const Matrix& x) const {
  validate_matrix(x);
  if (!features.rows() || x.cols() != features.cols())
    throw std::invalid_argument("Unfitted kNN or feature mismatch");
  Vector result(x.rows());
  for (Eigen::Index row = 0; row < x.rows(); ++row) {
    std::vector<std::pair<double, Eigen::Index>> distances;
    for (Eigen::Index i = 0; i < features.rows(); ++i)
      distances.emplace_back((features.row(i) - x.row(row)).squaredNorm(), i);
    std::partial_sort(distances.begin(), distances.begin() + neighbors, distances.end());
    double positive = 0;
    for (int i = 0; i < neighbors; ++i) positive += labels(distances[i].second);
    result(row) = positive / neighbors;
  }
  return result;
}
}  // namespace portfolio
