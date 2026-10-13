#include "portfolio/isolation_forest.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace portfolio {
static double adjustment(int size) {
  if (size <= 1)
    return 0;
  double harmonic = 0;
  for (int k = 1; k < size; ++k) harmonic += 1.0 / k;
  return 2 * harmonic - 2.0 * (size - 1) / size;
}
int IsolationForest::build(std::vector<IsolationNode>& tree, const Matrix& x,
                           const std::vector<Eigen::Index>& indices, int depth, int maximum,
                           std::mt19937& rng) {
  int node = tree.size();
  tree.push_back({-1, -1, -1, 0, static_cast<int>(indices.size())});
  if (indices.size() < 2 || depth >= maximum)
    return node;
  std::vector<int> features(features_);
  std::iota(features.begin(), features.end(), 0);
  std::shuffle(features.begin(), features.end(), rng);
  for (int feature : features) {
    double low = x(indices.front(), feature), high = low;
    for (auto row : indices) {
      low = std::min(low, x(row, feature));
      high = std::max(high, x(row, feature));
    }
    if (high <= low)
      continue;
    double cut = std::uniform_real_distribution<double>(low, high)(rng);
    std::vector<Eigen::Index> left, right;
    for (auto row : indices) (x(row, feature) < cut ? left : right).push_back(row);
    if (left.empty() || right.empty())
      continue;
    int l = build(tree, x, left, depth + 1, maximum, rng),
        r = build(tree, x, right, depth + 1, maximum, rng);
    tree[node] = {feature, l, r, cut, static_cast<int>(indices.size())};
    break;
  }
  return node;
}
void IsolationForest::fit(const Matrix& x, int trees, int sample_size, unsigned seed) {
  validate_matrix(x);
  if (trees < 1 || sample_size < 2 || x.rows() < 2)
    throw std::invalid_argument("Invalid isolation forest parameters");
  sample_size_ = std::min<int>(sample_size, x.rows());
  features_ = x.cols();
  trees_.clear();
  std::mt19937 rng(seed);
  std::vector<Eigen::Index> rows(x.rows());
  std::iota(rows.begin(), rows.end(), 0);
  for (int i = 0; i < trees; ++i) {
    std::shuffle(rows.begin(), rows.end(), rng);
    std::vector<Eigen::Index> selected(rows.begin(), rows.begin() + sample_size_);
    trees_.emplace_back();
    build(trees_.back(), x, selected, 0, std::ceil(std::log2(sample_size_)), rng);
  }
}
Vector IsolationForest::predict(const Matrix& x) const {
  validate_matrix(x);
  if (trees_.empty() || x.cols() != features_)
    throw std::invalid_argument("Unfitted forest or feature mismatch");
  Vector scores(x.rows());
  for (Eigen::Index row = 0; row < x.rows(); ++row) {
    double total = 0;
    for (const auto& tree : trees_) {
      int node = 0, depth = 0;
      while (tree[node].feature >= 0) {
        node = x(row, tree[node].feature) < tree[node].split ? tree[node].left : tree[node].right;
        ++depth;
      }
      total += depth + adjustment(tree[node].size);
    }
    scores(row) = std::pow(2.0, -total / (trees_.size() * adjustment(sample_size_)));
  }
  return scores;
}
Json IsolationForest::json() const {
  Json trees = Json::array();
  for (const auto& tree : trees_) {
    Json nodes = Json::array();
    for (const auto& n : tree) nodes.push_back({n.feature, n.left, n.right, n.split, n.size});
    trees.push_back(nodes);
  }
  return {{"kind", "isolation_forest"},
          {"features", features_},
          {"sample_size", sample_size_},
          {"trees", trees}};
}
IsolationForest IsolationForest::from_json(const Json& json) {
  if (json.at("kind") != "isolation_forest")
    throw std::invalid_argument("Unexpected model kind");
  IsolationForest model;
  model.features_ = json.at("features");
  model.sample_size_ = json.at("sample_size");
  if (model.features_ < 1 || model.sample_size_ < 2)
    throw std::invalid_argument("Invalid forest dimensions");
  for (const auto& values : json.at("trees")) {
    std::vector<IsolationNode> tree;
    for (const auto& n : values) {
      if (n.size() != 5)
        throw std::invalid_argument("Invalid forest node");
      tree.push_back(
          {n[0].get<int>(), n[1].get<int>(), n[2].get<int>(), n[3].get<double>(), n[4].get<int>()});
    }
    if (tree.empty())
      throw std::invalid_argument("Empty forest tree");
    for (std::size_t i = 0; i < tree.size(); ++i) {
      const auto& n = tree[i];
      if (!std::isfinite(n.split) || n.size < 1 || n.feature >= model.features_ ||
          (n.feature >= 0 &&
           (n.left <= static_cast<int>(i) || n.right <= static_cast<int>(i) ||
            n.left >= static_cast<int>(tree.size()) || n.right >= static_cast<int>(tree.size()))))
        throw std::invalid_argument("Invalid or cyclic forest tree");
    }
    model.trees_.push_back(std::move(tree));
  }
  if (model.trees_.empty())
    throw std::invalid_argument("Empty forest");
  return model;
}
RollingWindow::RollingWindow(std::size_t capacity, Eigen::Index features)
    : capacity_(capacity), features_(features) {
  if (capacity < 2 || features < 1)
    throw std::invalid_argument("Invalid window dimensions");
}
std::optional<Vector> RollingWindow::append(const Vector& reading) {
  if (reading.size() != features_ || !reading.allFinite())
    throw std::invalid_argument("Reading contract mismatch");
  readings_.push_back(reading);
  if (readings_.size() > capacity_)
    readings_.pop_front();
  if (readings_.size() < capacity_)
    return std::nullopt;
  Vector result(4 * features_);
  for (Eigen::Index feature = 0; feature < features_; ++feature) {
    double sum = 0, min = reading(feature), max = min;
    for (const auto& values : readings_) {
      sum += values(feature);
      min = std::min(min, values(feature));
      max = std::max(max, values(feature));
    }
    double mean = sum / capacity_, variance = 0;
    for (const auto& values : readings_) variance += std::pow(values(feature) - mean, 2);
    result.segment<4>(4 * feature) << mean, std::sqrt(variance / capacity_), reading(feature),
        max - min;
  }
  return result;
}
Dataset window_dataset(const Dataset& raw, std::size_t window) {
  validate_matrix(raw.x);
  if (raw.x.rows() < static_cast<Eigen::Index>(window) || raw.x.rows() != raw.y.size())
    throw std::invalid_argument("Insufficient aligned readings");
  RollingWindow state(window, raw.x.cols());
  std::deque<double> labels;
  Dataset result{
      Matrix(raw.x.rows() - window + 1, 4 * raw.x.cols()), Vector(raw.x.rows() - window + 1), {}};
  Eigen::Index index = 0;
  for (Eigen::Index i = 0; i < raw.x.rows(); ++i) {
    if (raw.y(i) != 0 && raw.y(i) != 1)
      throw std::invalid_argument("Binary window labels required");
    labels.push_back(raw.y(i));
    if (labels.size() > window)
      labels.pop_front();
    auto values = state.append(raw.x.row(i).transpose());
    if (values) {
      result.x.row(index) = values->transpose();
      result.y(index) = *std::max_element(labels.begin(), labels.end());
      ++index;
    }
  }
  for (const auto& name : raw.features)
    for (const auto& suffix : {"mean", "std", "last", "range"})
      result.features.push_back(name + "_" + suffix);
  return result;
}
}  // namespace portfolio
