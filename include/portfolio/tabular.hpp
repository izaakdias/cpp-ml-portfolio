#pragma once
#include "portfolio/data.hpp"
namespace portfolio {
struct LinearRegression {
  Vector weights;
  void fit(const Matrix& x, const Vector& y, double ridge = 1e-5);
  Vector predict(const Matrix& x) const;
};
struct LogisticRegression {
  Vector weights;
  void fit(const Matrix& x, const Vector& y, int epochs = 1500, double learning_rate = 0.05,
           double l2 = 0.001, bool balanced = false);
  Vector predict(const Matrix& x) const;
  Json json() const;
  static LogisticRegression from_json(const Json& json);
};
struct KNearestNeighbors {
  Matrix features;
  Vector labels;
  int neighbors = 5;
  void fit(const Matrix& x, const Vector& y, int k = 5);
  Vector predict(const Matrix& x) const;
};
}  // namespace portfolio
