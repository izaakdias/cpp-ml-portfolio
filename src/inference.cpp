#include "portfolio/inference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <variant>

#include "portfolio/registry.hpp"
namespace portfolio {
static std::string type_name(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    default:
      return "unsupported:" + std::to_string(type);
  }
}
InferenceEngine::InferenceEngine(const std::filesystem::path& model, int threads,
                                 const std::string& provider)
    : environment_(ORT_LOGGING_LEVEL_WARNING, "cpp-ml-portfolio"),
      checksum_(sha256(model)),
      provider_(provider) {
  if (threads < 1 || threads > 64)
    throw std::invalid_argument("Threads must be 1..64");
  options_.SetIntraOpNumThreads(threads);
  options_.SetInterOpNumThreads(1);
  options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (provider == "CoreML")
    options_.AppendExecutionProvider("CoreML",
                                     {{"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"}});
  else if (provider != "CPU")
    throw std::invalid_argument("Provider must be CPU or CoreML");
  session_ = Ort::Session(environment_, model.c_str(), options_);
  Ort::AllocatorWithDefaultOptions allocator;
  for (std::size_t i = 0; i < session_.GetInputCount(); ++i)
    inputs_.emplace_back(session_.GetInputNameAllocated(i, allocator).get());
  for (std::size_t i = 0; i < session_.GetOutputCount(); ++i)
    outputs_.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
}
Json InferenceEngine::describe() const {
  Json inputs = Json::array(), outputs = Json::array();
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    auto type = session_.GetInputTypeInfo(i);
    auto info = type.GetTensorTypeAndShapeInfo();
    inputs.push_back({{"name", inputs_[i]},
                      {"dtype", type_name(info.GetElementType())},
                      {"shape", info.GetShape()}});
  }
  for (std::size_t i = 0; i < outputs_.size(); ++i) {
    auto type = session_.GetOutputTypeInfo(i);
    auto info = type.GetTensorTypeAndShapeInfo();
    outputs.push_back({{"name", outputs_[i]},
                       {"dtype", type_name(info.GetElementType())},
                       {"shape", info.GetShape()}});
  }
  return {{"model_sha256", checksum_},
          {"requested_provider", provider_},
          {"inputs", inputs},
          {"outputs", outputs}};
}
Json InferenceEngine::run(const Json& inputs) {
  if (!inputs.is_object() || inputs.size() != inputs_.size())
    throw std::invalid_argument("Input names must exactly match the model");
  using Storage =
      std::variant<std::vector<float>, std::vector<std::int64_t>, std::vector<std::uint8_t>>;
  std::vector<Storage> buffers;
  std::vector<Ort::Value> tensors;
  std::vector<const char*> names, output_names;
  buffers.reserve(inputs_.size());
  tensors.reserve(inputs_.size());
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  for (std::size_t i = 0; i < inputs_.size(); ++i) {
    if (!inputs.contains(inputs_[i]))
      throw std::invalid_argument("Missing model input: " + inputs_[i]);
    const auto& input = inputs.at(inputs_[i]);
    if (!input.is_object() || input.size() != 3 || !input.contains("shape") ||
        !input.contains("dtype") || !input.contains("data"))
      throw std::invalid_argument("Tensor requires exactly dtype,shape,data");
    if (!input.at("shape").is_array())
      throw std::invalid_argument("Shape must be an integer array");
    for (const auto& dimension : input.at("shape"))
      if (!dimension.is_number_integer())
        throw std::invalid_argument("Integer dimensions required");
    auto shape = input.at("shape").get<std::vector<std::int64_t>>();
    auto type = session_.GetInputTypeInfo(i);
    auto info = type.GetTensorTypeAndShapeInfo();
    auto expected = info.GetShape();
    if (shape.size() != expected.size() || input.at("dtype") != type_name(info.GetElementType()))
      throw std::invalid_argument("Tensor rank or dtype differs from model");
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
      if (shape[axis] < 1 || shape[axis] > 10000000 ||
          count > 10000000 / static_cast<std::size_t>(shape[axis]) ||
          (expected[axis] >= 0 && expected[axis] != shape[axis]))
        throw std::invalid_argument("Invalid tensor shape");
      count *= shape[axis];
    }
    const auto& values = input.at("data");
    if (!values.is_array() || values.size() != count)
      throw std::invalid_argument("Tensor data count differs from shape");
    for (const auto& value : values)
      if (!value.is_number() || !std::isfinite(value.get<double>()))
        throw std::invalid_argument("Finite numeric tensor values required");
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      std::vector<float> data;
      data.reserve(count);
      for (const auto& value : values) {
        auto number = value.get<float>();
        if (!std::isfinite(number))
          throw std::invalid_argument("Float32 overflow");
        data.push_back(number);
      }
      buffers.emplace_back(std::move(data));
      auto& buffer = std::get<std::vector<float>>(buffers.back());
      tensors.push_back(Ort::Value::CreateTensor<float>(memory, buffer.data(), buffer.size(),
                                                        shape.data(), shape.size()));
    } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      std::vector<std::int64_t> data;
      data.reserve(count);
      for (const auto& value : values) {
        if (!value.is_number_integer() ||
            (value.is_number_unsigned() &&
             value.get<std::uint64_t>() > static_cast<std::uint64_t>(INT64_MAX)))
          throw std::invalid_argument("Int64 integers required");
        data.push_back(value.get<std::int64_t>());
      }
      buffers.emplace_back(std::move(data));
      auto& buffer = std::get<std::vector<std::int64_t>>(buffers.back());
      tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(memory, buffer.data(), buffer.size(),
                                                               shape.data(), shape.size()));
    } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      std::vector<std::uint8_t> data;
      data.reserve(count);
      for (const auto& value : values) {
        if (!value.is_number_integer() || value.get<double>() < 0 || value.get<double>() > 255)
          throw std::invalid_argument("Uint8 integers in 0..255 required");
        data.push_back(value.get<std::uint8_t>());
      }
      buffers.emplace_back(std::move(data));
      auto& buffer = std::get<std::vector<std::uint8_t>>(buffers.back());
      tensors.push_back(Ort::Value::CreateTensor<std::uint8_t>(memory, buffer.data(), buffer.size(),
                                                               shape.data(), shape.size()));
    } else
      throw std::invalid_argument("Unsupported input dtype");
    names.push_back(inputs_[i].c_str());
  }
  for (const auto& name : outputs_) output_names.push_back(name.c_str());
  auto results = session_.Run(Ort::RunOptions{nullptr}, names.data(), tensors.data(),
                              tensors.size(), output_names.data(), output_names.size());
  Json output = Json::object();
  for (std::size_t i = 0; i < results.size(); ++i) {
    if (!results[i].IsTensor())
      throw std::runtime_error("Only tensor outputs are supported");
    auto info = results[i].GetTensorTypeAndShapeInfo();
    auto count = info.GetElementCount();
    Json values;
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      auto data = results[i].GetTensorData<float>();
      for (std::size_t index = 0; index < count; ++index)
        if (!std::isfinite(data[index]))
          throw std::runtime_error("Model returned nonfinite output");
      values = std::vector<float>(data, data + count);
    } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      auto data = results[i].GetTensorData<std::int64_t>();
      values = std::vector<std::int64_t>(data, data + count);
    } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      auto data = results[i].GetTensorData<std::uint8_t>();
      values = std::vector<std::uint8_t>(data, data + count);
    } else
      throw std::runtime_error("Unsupported output dtype");
    output[outputs_[i]] = {
        {"dtype", type_name(info.GetElementType())}, {"shape", info.GetShape()}, {"data", values}};
  }
  return {{"model_sha256", checksum_}, {"outputs", output}};
}
Json benchmark(InferenceEngine& engine, const Json& inputs, int warmup, int repetitions) {
  if (warmup < 0 || repetitions < 10 || repetitions > 100000)
    throw std::invalid_argument("Benchmark requires 10..100000 repetitions");
  for (int i = 0; i < warmup; ++i) engine.run(inputs);
  std::vector<double> times;
  times.reserve(repetitions);
  for (int i = 0; i < repetitions; ++i) {
    auto begin = std::chrono::steady_clock::now();
    engine.run(inputs);
    times.push_back(
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
            .count());
  }
  std::sort(times.begin(), times.end());
  auto percentile = [&](double p) {
    return times[static_cast<std::size_t>(std::ceil(p * times.size())) - 1];
  };
  return {{"model_sha256", engine.checksum()},
          {"scope", "validation_tensor_allocation_inference_output_copy"},
          {"warmup", warmup},
          {"samples", repetitions},
          {"p50_ms", percentile(0.5)},
          {"p95_ms", percentile(0.95)},
          {"p99_ms", percentile(0.99)}};
}
}  // namespace portfolio
