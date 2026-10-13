#pragma once
#include "portfolio/data.hpp"
namespace portfolio {
struct Artifact {
  Json model, metadata;
  std::filesystem::path path;
};
std::string sha256(const std::filesystem::path& path);
Artifact save_artifact(const std::filesystem::path& root, const std::string& project,
                       const Json& model, const Json& metadata);
Artifact load_artifact(const std::filesystem::path& root, const std::string& project,
                       const std::string& run_id = "");
void promote_artifact(const std::filesystem::path& root, const std::string& project,
                      const std::string& run_id);
void write_json(const std::filesystem::path& path, const Json& value);
Json read_json(const std::filesystem::path& path);
}  // namespace portfolio
