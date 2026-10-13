#include <catch2/catch_test_macros.hpp>
#include <future>
#include <set>

#include "portfolio/registry.hpp"
#include "support.hpp"
using namespace portfolio;
TEST_CASE("Registry roundtrip and explicit promotion") {
  TempDirectory directory;
  auto first = save_artifact(directory.path, "demo", {{"value", 1}}, {});
  auto second = save_artifact(directory.path, "demo", {{"value", 2}}, {});
  REQUIRE(load_artifact(directory.path, "demo").model["value"] == 2);
  promote_artifact(directory.path, "demo", first.metadata["run_id"]);
  REQUIRE(load_artifact(directory.path, "demo").model["value"] == 1);
  REQUIRE(load_artifact(directory.path, "demo", second.metadata["run_id"]).model["value"] == 2);
}
TEST_CASE("Registry detects accidental corruption") {
  TempDirectory directory;
  auto artifact = save_artifact(directory.path, "demo", {{"value", 1}}, {});
  std::ofstream(artifact.path / "model.json") << "{}";
  REQUIRE_THROWS(load_artifact(directory.path, "demo"));
}
TEST_CASE("Failed serialization preserves current model") {
  TempDirectory directory;
  auto original = save_artifact(directory.path, "demo", {}, {});
  REQUIRE_THROWS(save_artifact(directory.path, "demo",
                               {{"value", std::numeric_limits<double>::infinity()}}, {}));
  REQUIRE(load_artifact(directory.path, "demo").metadata["run_id"] == original.metadata["run_id"]);
}
TEST_CASE("Registry prevents path traversal") {
  TempDirectory directory;
  REQUIRE_THROWS(save_artifact(directory.path, "../escape", {}, {}));
  REQUIRE_THROWS(load_artifact(directory.path, "demo", "../escape"));
}
TEST_CASE("Concurrent registry writers publish complete unique versions") {
  TempDirectory directory;
  std::vector<std::future<Artifact>> futures;
  for (int i = 0; i < 8; ++i)
    futures.push_back(std::async(std::launch::async, [&, i] {
      return save_artifact(directory.path, "demo", {{"value", i}}, {});
    }));
  std::set<std::string> ids;
  for (auto& future : futures) {
    auto artifact = future.get();
    REQUIRE(ids.insert(artifact.metadata["run_id"]).second);
    REQUIRE(load_artifact(directory.path, "demo", artifact.metadata["run_id"]).model ==
            artifact.model);
  }
  REQUIRE(
      ids.contains(load_artifact(directory.path, "demo").metadata["run_id"].get<std::string>()));
}
TEST_CASE("SHA256 matches a known digest") {
  TempDirectory directory;
  REQUIRE(sha256(directory.file("message", "abc")) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}
