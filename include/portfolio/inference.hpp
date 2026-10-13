#pragma once
#include <onnxruntime_cxx_api.h>

#include <mutex>

#include "portfolio/data.hpp"
namespace portfolio {
class InferenceEngine {
  Ort::Env environment_;
  Ort::SessionOptions options_;
  Ort::Session session_{nullptr};
  std::vector<std::string> inputs_, outputs_;
  std::string checksum_, provider_;

 public:
  InferenceEngine(const std::filesystem::path& model, int threads = 1,
                  const std::string& provider = "CPU");
  Json describe() const;
  Json run(const Json& inputs);
  const std::string& checksum() const {
    return checksum_;
  }
};
Json benchmark(InferenceEngine& engine, const Json& inputs, int warmup = 10, int repetitions = 100);
}  // namespace portfolio
