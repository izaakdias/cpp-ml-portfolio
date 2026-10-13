#include <catch2/catch_test_macros.hpp>
#include <random>

#include "portfolio/clustering.hpp"
#include "portfolio/isolation_forest.hpp"
#include "portfolio/recommendation.hpp"
#include "portfolio/tabular.hpp"
using namespace portfolio;
TEST_CASE("Linear regression recovers a known affine function") {
  Matrix x(10, 2);
  Vector y(10);
  for (int i = 0; i < 10; ++i) {
    x(i, 0) = i;
    x(i, 1) = i * i;
    y(i) = 2 + 3 * x(i, 0) - 0.2 * x(i, 1);
  }
  LinearRegression model;
  model.fit(x, y, 0);
  REQUIRE(model.predict(x).isApprox(y, 1e-10));
}
TEST_CASE("Logistic learns separable labels and survives serialization") {
  Matrix x(6, 1);
  x << -3, -2, -1, 1, 2, 3;
  Vector y(6);
  y << 0, 0, 0, 1, 1, 1;
  LogisticRegression model;
  model.fit(x, y);
  auto p = model.predict(x);
  REQUIRE(p(0) < 0.1);
  REQUIRE(p(5) > 0.9);
  REQUIRE(LogisticRegression::from_json(model.json()).predict(x).isApprox(p));
  REQUIRE_THROWS(model.fit(x, Vector::Zero(6)));
}
TEST_CASE("kNN classifies nearby points") {
  Matrix x(4, 1);
  x << 0, 0.1, 2, 2.1;
  Vector y(4);
  y << 0, 0, 1, 1;
  KNearestNeighbors model;
  model.fit(x, y, 1);
  REQUIRE(model.predict(x) == y);
  REQUIRE_THROWS(model.fit(x, y, 5));
}
TEST_CASE("Clustering algorithms recover separated groups") {
  Matrix x(12, 2);
  Vector truth(12);
  for (int i = 0; i < 12; ++i) {
    x(i, 0) = (i < 6 ? -5 : 5) + 0.02 * i;
    x(i, 1) = 0.03 * i;
    truth(i) = i < 6 ? 0 : 1;
  }
  REQUIRE(adjusted_rand(truth, kmeans(x, 2).labels) > 0.99);
  REQUIRE(adjusted_rand(truth, dbscan(x, 0.5, 3)) > 0.99);
  REQUIRE(adjusted_rand(truth, gaussian_mixture(x, 2)) > 0.99);
  REQUIRE(pca(x).rows() == 12);
}
TEST_CASE("DBSCAN marks isolated observations as noise") {
  Matrix x(4, 1);
  x << 0, 0.1, 0.2, 100;
  auto labels = dbscan(x, 0.3, 2);
  REQUIRE(labels(3) == -1);
  REQUIRE(labels(0) == labels(2));
}
TEST_CASE("Isolation forest ranks an extreme point above normal data") {
  std::mt19937 rng(1);
  std::normal_distribution<double> normal;
  Matrix x(200, 2);
  for (auto& value : x.reshaped()) value = normal(rng);
  IsolationForest model;
  model.fit(x, 100);
  Matrix probe(2, 2);
  probe << 0, 0, 20, 20;
  auto scores = model.predict(probe);
  REQUIRE(scores(1) > scores(0));
  REQUIRE(IsolationForest::from_json(model.json()).predict(probe) == scores);
}
TEST_CASE("Rolling window and batch features match") {
  Dataset raw{Matrix(30, 1), Vector::Zero(30), {"sensor"}};
  for (int i = 0; i < 30; ++i) raw.x(i, 0) = i;
  raw.y(5) = 1;
  auto batch = window_dataset(raw);
  RollingWindow stream(24, 1);
  for (int i = 0; i < 30; ++i) {
    auto result = stream.append(raw.x.row(i).transpose());
    if (result)
      REQUIRE(result->isApprox(batch.x.row(i - 23).transpose()));
  }
  REQUIRE(batch.y(0) == 1);
  REQUIRE(batch.y(6) == 0);
}
TEST_CASE("Latest interaction holdout retains training history") {
  std::vector<Rating> ratings = {
      {"a", "1", 5, 1}, {"a", "2", 4, 2}, {"b", "2", 5, 1}, {"b", "3", 3, 3}};
  auto [train, test] = holdout_latest(ratings);
  REQUIRE(train.size() == 2);
  REQUIRE(test.size() == 2);
  REQUIRE(test[0].item == "2");
}
TEST_CASE("Factorization excludes seen items") {
  std::vector<Rating> ratings = {
      {"a", "1", 5, 1}, {"a", "2", 4, 2}, {"b", "2", 5, 1}, {"b", "3", 3, 3}};
  MatrixFactorization model;
  model.fit(ratings);
  auto result = model.recommend("a");
  REQUIRE(result.size() == 1);
  REQUIRE(result[0]["item_id"] == "3");
  REQUIRE_THROWS(model.recommend("unknown"));
}
