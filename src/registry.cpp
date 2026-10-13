#include "portfolio/registry.hpp"

#include <openssl/evp.h>

#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
namespace portfolio {
static std::string safe(const std::string& value) {
  if (!std::regex_match(value, std::regex("[A-Za-z0-9][A-Za-z0-9_.-]*")))
    throw std::invalid_argument("Invalid registry name");
  return value;
}
static std::string identifier() {
  std::random_device random;
  std::ostringstream stream;
  stream << std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count()
         << '-' << std::hex << random() << random();
  return stream.str();
}
static void finite_json(const Json& value) {
  if (value.is_number_float() && !std::isfinite(value.get<double>()))
    throw std::invalid_argument("Nonfinite JSON value");
  if (value.is_array() || value.is_object())
    for (const auto& item : value) finite_json(item);
}
void write_json(const std::filesystem::path& path, const Json& value) {
  finite_json(value);
  std::ofstream output(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("Cannot write JSON: " + path.string());
  output << value.dump(2) << '\n';
  output.flush();
  if (!output)
    throw std::runtime_error("JSON write failed");
}
Json read_json(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("Cannot read JSON: " + path.string());
  Json value;
  input >> value;
  return value;
}
std::string sha256(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("Cannot checksum file");
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(),
                                                                  EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("SHA-256 initialization failed");
  std::array<char, 65536> block;
  while (input) {
    input.read(block.data(), block.size());
    if (EVP_DigestUpdate(context.get(), block.data(), input.gcount()) != 1)
      throw std::runtime_error("SHA-256 update failed");
  }
  if (!input.eof())
    throw std::runtime_error("Checksum read failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest;
  unsigned length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1)
    throw std::runtime_error("SHA-256 finalization failed");
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (unsigned i = 0; i < length; ++i) result << std::setw(2) << static_cast<unsigned>(digest[i]);
  return result.str();
}
static void pointer(const std::filesystem::path& root, const std::string& id) {
  auto temporary = root / (".latest-" + identifier() + ".json");
  try {
    write_json(temporary, {{"run_id", id}});
    std::filesystem::rename(temporary, root / "latest.json");
  } catch (...) {
    std::error_code error;
    std::filesystem::remove(temporary, error);
    throw;
  }
}
Artifact save_artifact(const std::filesystem::path& root, const std::string& project,
                       const Json& model, const Json& metadata) {
  auto project_root = root / safe(project);
  std::filesystem::create_directories(project_root);
  auto id = identifier();
  auto temporary = project_root / (".staging-" + id);
  auto final = project_root / id;
  std::filesystem::create_directory(temporary);
  Json saved = metadata;
  saved["format_version"] = 1;
  saved["project"] = project;
  saved["run_id"] = id;
  try {
    write_json(temporary / "model.json", model);
    write_json(temporary / "metadata.json", saved);
    write_json(temporary / "manifest.json",
               {{"model.json", sha256(temporary / "model.json")},
                {"metadata.json", sha256(temporary / "metadata.json")}});
    std::filesystem::rename(temporary, final);
    pointer(project_root, id);
  } catch (...) {
    std::error_code error;
    std::filesystem::remove_all(temporary, error);
    throw;
  }
  return {model, saved, final};
}
Artifact load_artifact(const std::filesystem::path& root, const std::string& project,
                       const std::string& run_id) {
  auto project_root = root / safe(project);
  std::string id = run_id.empty()
                       ? read_json(project_root / "latest.json").at("run_id").get<std::string>()
                       : run_id;
  auto path = project_root / safe(id);
  auto manifest = read_json(path / "manifest.json");
  if (!manifest.contains("model.json") || !manifest.contains("metadata.json"))
    throw std::invalid_argument("Incomplete artifact manifest");
  for (auto it = manifest.begin(); it != manifest.end(); ++it)
    if (sha256(path / safe(it.key())) != it.value().get<std::string>())
      throw std::runtime_error("Artifact integrity mismatch");
  auto metadata = read_json(path / "metadata.json");
  if (metadata.at("format_version") != 1 || metadata.at("project") != project ||
      metadata.at("run_id") != id)
    throw std::runtime_error("Artifact metadata mismatch");
  return {read_json(path / "model.json"), metadata, path};
}
void promote_artifact(const std::filesystem::path& root, const std::string& project,
                      const std::string& run_id) {
  load_artifact(root, project, run_id);
  pointer(root / safe(project), safe(run_id));
}
}  // namespace portfolio
