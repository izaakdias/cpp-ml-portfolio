#pragma once
#include "portfolio/data.hpp"
namespace portfolio {
struct Clusters {
  Matrix centers;
  Eigen::VectorXi labels;
  double objective = 0;
};
Clusters kmeans(const Matrix& x, int k, unsigned seed = 41, int iterations = 100);
Eigen::VectorXi dbscan(const Matrix& x, double epsilon, int minimum_points);
Eigen::VectorXi gaussian_mixture(const Matrix& x, int k, unsigned seed = 41, int iterations = 100);
Matrix pca(const Matrix& x, int components = 2);
double adjusted_rand(const Vector& truth, const Eigen::VectorXi& clusters);
}  // namespace portfolio
