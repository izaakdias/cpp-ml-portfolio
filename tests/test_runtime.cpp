#include <catch2/catch_test_macros.hpp>

#include "portfolio/inference.hpp"
#include "portfolio/tokenizer.hpp"
#include "support.hpp"
using namespace portfolio;
TEST_CASE("ONNX runtime executes dynamic batches") {
  InferenceEngine engine(PORTFOLIO_TEST_MODEL);
  auto result =
      engine.run({{"features", {{"dtype", "float32"}, {"shape", {2, 2}}, {"data", {1, 2, 3, 4}}}}});
  auto values = result["outputs"]["prediction"]["data"].get<std::vector<float>>();
  REQUIRE(values == std::vector<float>{0.5f, 2.5f});
}
TEST_CASE("ONNX runtime rejects dtype shape names and nonfinite data") {
  InferenceEngine engine(PORTFOLIO_TEST_MODEL);
  REQUIRE_THROWS(
      engine.run({{"wrong", {{"dtype", "float32"}, {"shape", {1, 2}}, {"data", {1, 2}}}}}));
  REQUIRE_THROWS(
      engine.run({{"features", {{"dtype", "int64"}, {"shape", {1, 2}}, {"data", {1, 2}}}}}));
  REQUIRE_THROWS(
      engine.run({{"features", {{"dtype", "float32"}, {"shape", {1, 3}}, {"data", {1, 2, 3}}}}}));
  REQUIRE_THROWS(engine.run({{"features",
                              {{"dtype", "float32"},
                               {"shape", {1, 2}},
                               {"data", {std::numeric_limits<double>::quiet_NaN(), 2}}}}}));
}
TEST_CASE("WordPiece normalizes accents and applies greedy subwords") {
  TempDirectory directory;
  auto path = directory.file("vocab", "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ncafe\nplay\n##ing\n!\n");
  WordPieceTokenizer tokenizer(path);
  auto result = tokenizer.encode("CAFÉ playing!", 8);
  REQUIRE(result.ids == std::vector<std::int64_t>{2, 5, 6, 7, 8, 3, 0, 0});
  REQUIRE(result.mask == std::vector<std::int64_t>{1, 1, 1, 1, 1, 1, 0, 0});
}
TEST_CASE("WordPiece truncates with SEP and handles special tokens") {
  TempDirectory directory;
  auto path = directory.file("vocab", "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n");
  WordPieceTokenizer tokenizer(path);
  REQUIRE(tokenizer.encode("hello hello hello", 3).ids == std::vector<std::int64_t>{2, 5, 3});
  REQUIRE(tokenizer.encode("[MASK] hello", 5).ids == std::vector<std::int64_t>{2, 4, 5, 3, 0});
  REQUIRE_THROWS(tokenizer.encode("hello", 1));
}
