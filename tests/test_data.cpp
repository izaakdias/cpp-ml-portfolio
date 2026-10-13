#include <catch2/catch_test_macros.hpp>
#include <set>

#include "portfolio/data.hpp"
#include "portfolio/metrics.hpp"
#include "support.hpp"
using namespace portfolio;
TEST_CASE("CSV handles quotes commas and embedded newlines") {
  TempDirectory directory;
  auto path = directory.file("a.csv", "id,text\r\n1,\"hello, world\"\r\n2,\"line\nnext\"\r\n");
  auto rows = read_csv(path);
  REQUIRE(rows.size() == 3);
  REQUIRE(rows[1][1] == "hello, world");
  REQUIRE(rows[2][1] == "line\nnext");
}
TEST_CASE("CSV rejects inconsistent columns and malformed quoting") {
  TempDirectory directory;
  REQUIRE_THROWS(read_csv(directory.file("a.csv", "a,b\n1\n")));
  REQUIRE_THROWS(read_csv(directory.file("b.csv", "a,b\n1,\"open")));
}
TEST_CASE("Numeric ingestion rejects nonfinite and trailing data") {
  REQUIRE_THROWS(number("nan"));
  REQUIRE_THROWS(number("inf"));
  REQUIRE_THROWS(number("12garbage"));
  REQUIRE(number("1.25") == 1.25);
}
TEST_CASE("Standardization handles constant columns and roundtrip") {
  Matrix x(3, 2);
  x << 1, 4, 2, 4, 3, 4;
  auto scaler = Standardizer::fit(x);
  auto normalized = scaler.transform(x);
  REQUIRE(normalized.col(1).isZero());
  REQUIRE(Standardizer::from_json(scaler.json()).transform(x).isApprox(normalized));
}
TEST_CASE("Splits are repeatable and disjoint") {
  Dataset data{Matrix(20, 1), Vector(20), {"id"}};
  for (int i = 0; i < 20; ++i) {
    data.x(i, 0) = i;
    data.y(i) = i % 2;
  }
  auto a = split_dataset(data, 41, true), b = split_dataset(data, 41, true);
  REQUIRE(a.train.x == b.train.x);
  std::set<int> ids;
  for (const auto* part : {&a.train, &a.validation, &a.test})
    for (Eigen::Index i = 0; i < part->x.rows(); ++i) REQUIRE(ids.insert(part->x(i, 0)).second);
  REQUIRE(ids.size() == 20);
}
TEST_CASE("Metrics handle tied scores and undefined classes") {
  Vector y(4), scores(4);
  y << 0, 1, 0, 1;
  scores.setConstant(0.5);
  auto metrics = binary_metrics(y, scores, 0.5);
  REQUIRE(metrics["roc_auc"] == 0.5);
  REQUIRE(metrics["average_precision"] == 0.5);
  REQUIRE(binary_metrics(Vector::Zero(4), scores, 0.5)["roc_auc"].is_null());
}
