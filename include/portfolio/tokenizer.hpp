#pragma once
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>
namespace portfolio {
struct EncodedText {
  std::vector<std::int64_t> ids, mask;
};
class WordPieceTokenizer {
  std::unordered_map<std::string, std::int64_t> vocabulary_;
  std::vector<std::string> words(const std::string& text) const;
  std::vector<std::int64_t> pieces(const std::string& word) const;

 public:
  explicit WordPieceTokenizer(const std::filesystem::path& vocabulary);
  EncodedText encode(const std::string& text, std::size_t maximum = 128) const;
};
}  // namespace portfolio
