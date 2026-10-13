#pragma once
#include <deque>
#include <optional>
#include <random>

#include "portfolio/data.hpp"
namespace portfolio {
struct IsolationNode {
  int feature = -1, left = -1, right = -1;
  double split = 0;
  int size = 0;
};
class IsolationForest {
  std::vector<std::vector<IsolationNode>> trees_;
  int sample_size_ = 0, features_ = 0;
  int build(std::vector<IsolationNode>& tree, const Matrix& x,
            const std::vector<Eigen::Index>& indices, int depth, int maximum, std::mt19937& rng);

 public:
  void fit(const Matrix& x, int trees = 180, int sample_size = 256, unsigned seed = 71);
  Vector predict(const Matrix& x) const;
  Json json() const;
  static IsolationForest from_json(const Json& json);
};
class RollingWindow {
  std::deque<Vector> readings_;
  std::size_t capacity_;
  Eigen::Index features_;

 public:
  RollingWindow(std::size_t capacity, Eigen::Index features);
  std::optional<Vector> append(const Vector& reading);
};
Dataset window_dataset(const Dataset& raw, std::size_t window = 24);
}  // namespace portfolio
