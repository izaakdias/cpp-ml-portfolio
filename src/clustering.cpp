#include "portfolio/clustering.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <random>
#include <stdexcept>
namespace portfolio {
Clusters kmeans(const Matrix& x, int k, unsigned seed, int iterations) {
  validate_matrix(x);
  if (k < 1 || k > x.rows() || iterations < 1)
    throw std::invalid_argument("Invalid K-means parameters");
  std::mt19937 rng(seed);
  Matrix centers(k, x.cols());
  std::uniform_int_distribution<Eigen::Index> random_row(0, x.rows() - 1);
  centers.row(0) = x.row(random_row(rng));
  Vector distances = Vector::Constant(x.rows(), std::numeric_limits<double>::infinity());
  for (int center = 1; center < k; ++center) {
    for (Eigen::Index i = 0; i < x.rows(); ++i)
      distances(i) = std::min(distances(i), (x.row(i) - centers.row(center - 1)).squaredNorm());
    std::discrete_distribution<Eigen::Index> choose(distances.data(),
                                                    distances.data() + distances.size());
    centers.row(center) = x.row(choose(rng));
  }
  Eigen::VectorXi labels = Eigen::VectorXi::Constant(x.rows(), -1);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    auto old = labels;
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      double best = std::numeric_limits<double>::infinity();
      for (int j = 0; j < k; ++j) {
        double distance = (x.row(i) - centers.row(j)).squaredNorm();
        if (distance < best) {
          best = distance;
          labels(i) = j;
        }
      }
    }
    Matrix sum = Matrix::Zero(k, x.cols());
    Eigen::VectorXi count = Eigen::VectorXi::Zero(k);
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      sum.row(labels(i)) += x.row(i);
      ++count(labels(i));
    }
    for (int j = 0; j < k; ++j)
      if (count(j))
        centers.row(j) = sum.row(j) / count(j);
      else
        centers.row(j) = x.row(random_row(rng));
    if (labels == old)
      break;
  }
  double objective = 0;
  for (Eigen::Index i = 0; i < x.rows(); ++i)
    objective += (x.row(i) - centers.row(labels(i))).squaredNorm();
  return {centers, labels, objective};
}
Eigen::VectorXi dbscan(const Matrix& x, double epsilon, int minimum_points) {
  validate_matrix(x);
  if (!std::isfinite(epsilon) || epsilon <= 0 || minimum_points < 1)
    throw std::invalid_argument("Invalid DBSCAN parameters");
  Eigen::VectorXi labels = Eigen::VectorXi::Constant(x.rows(), -2);
  auto neighbors = [&](Eigen::Index row) {
    std::vector<Eigen::Index> found;
    for (Eigen::Index i = 0; i < x.rows(); ++i)
      if ((x.row(row) - x.row(i)).squaredNorm() <= epsilon * epsilon)
        found.push_back(i);
    return found;
  };
  int group = 0;
  for (Eigen::Index i = 0; i < x.rows(); ++i) {
    if (labels(i) != -2)
      continue;
    auto queue = neighbors(i);
    if (queue.size() < static_cast<std::size_t>(minimum_points)) {
      labels(i) = -1;
      continue;
    }
    labels(i) = group;
    std::vector<bool> queued(x.rows(), false);
    for (auto row : queue) queued[row] = true;
    for (std::size_t pos = 0; pos < queue.size(); ++pos) {
      auto row = queue[pos];
      if (labels(row) == -1)
        labels(row) = group;
      if (labels(row) != -2)
        continue;
      labels(row) = group;
      auto expanded = neighbors(row);
      if (expanded.size() >= static_cast<std::size_t>(minimum_points))
        for (auto item : expanded)
          if (!queued[item]) {
            queued[item] = true;
            queue.push_back(item);
          }
    }
    ++group;
  }
  return labels;
}
Eigen::VectorXi gaussian_mixture(const Matrix& x, int k, unsigned seed, int iterations) {
  if (iterations < 1)
    throw std::invalid_argument("Positive EM iterations required");
  auto initial = kmeans(x, k, seed);
  Matrix means = initial.centers;
  Vector global_variance = (x.rowwise() - x.colwise().mean()).array().square().colwise().mean();
  Matrix variance = global_variance.transpose().replicate(k, 1).array().max(1e-6);
  Vector weights = Vector::Constant(k, 1.0 / k);
  Matrix responsibilities(x.rows(), k);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    for (Eigen::Index i = 0; i < x.rows(); ++i) {
      Vector log_probability(k);
      for (int j = 0; j < k; ++j)
        log_probability(j) =
            std::log(std::max(weights(j), 1e-12)) -
            0.5 * (variance.row(j).array().log() +
                   ((x.row(i) - means.row(j)).array().square() / variance.row(j).array()) +
                   std::log(2 * std::numbers::pi))
                      .sum();
      Vector stable = (log_probability.array() - log_probability.maxCoeff()).exp();
      responsibilities.row(i) = (stable / stable.sum()).transpose();
    }
    for (int j = 0; j < k; ++j) {
      double count = responsibilities.col(j).sum();
      if (count < 1e-8)
        continue;
      means.row(j) = (responsibilities.col(j).transpose() * x) / count;
      variance.row(j) = ((x.rowwise() - means.row(j)).array().square().colwise() *
                         responsibilities.col(j).array())
                            .colwise()
                            .sum() /
                        count;
      variance.row(j) = variance.row(j).array().max(1e-6);
      weights(j) = count / x.rows();
    }
    weights /= weights.sum();
  }
  Eigen::VectorXi labels(x.rows());
  for (Eigen::Index i = 0; i < x.rows(); ++i) {
    Eigen::Index maximum;
    responsibilities.row(i).maxCoeff(&maximum);
    labels(i) = maximum;
  }
  return labels;
}
Matrix pca(const Matrix& x, int components) {
  validate_matrix(x);
  if (x.rows() < 2 || components < 1 || components > x.cols())
    throw std::invalid_argument("Invalid PCA dimensions");
  Matrix centered = x.rowwise() - x.colwise().mean();
  Matrix covariance = centered.transpose() * centered / (x.rows() - 1);
  Eigen::SelfAdjointEigenSolver<Matrix> solver(covariance);
  if (solver.info() != Eigen::Success)
    throw std::runtime_error("PCA decomposition failed");
  Matrix axes(x.cols(), components);
  for (int component = 0; component < components; ++component)
    axes.col(component) = solver.eigenvectors().col(x.cols() - 1 - component);
  return centered * axes;
}
double adjusted_rand(const Vector& truth, const Eigen::VectorXi& clusters) {
  if (truth.size() != clusters.size() || truth.size() < 2 || !truth.allFinite())
    throw std::invalid_argument("Aligned labels required");
  std::map<double, int> rows;
  std::map<int, int> columns;
  std::map<std::pair<double, int>, int> cells;
  for (Eigen::Index i = 0; i < truth.size(); ++i) {
    ++rows[truth(i)];
    ++columns[clusters(i)];
    ++cells[{truth(i), clusters(i)}];
  }
  auto pairs = [](double n) { return n * (n - 1) / 2; };
  double a = 0, b = 0, c = 0;
  for (auto [key, count] : rows) {
    (void)key;
    a += pairs(count);
  }
  for (auto [key, count] : columns) {
    (void)key;
    b += pairs(count);
  }
  for (auto [key, count] : cells) {
    (void)key;
    c += pairs(count);
  }
  double expected = a * b / pairs(truth.size()), denominator = (a + b) / 2 - expected;
  return std::abs(denominator) < 1e-12 ? 1.0 : (c - expected) / denominator;
}
}  // namespace portfolio
