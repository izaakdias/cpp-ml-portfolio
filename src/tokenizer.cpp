#include "portfolio/tokenizer.hpp"

#include <utf8proc.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
namespace portfolio {
WordPieceTokenizer::WordPieceTokenizer(const std::filesystem::path& vocabulary) {
  std::ifstream input(vocabulary);
  if (!input)
    throw std::runtime_error("Cannot open vocabulary");
  std::string word;
  std::int64_t index = 0;
  while (std::getline(input, word)) {
    if (!word.empty() && word.back() == '\r')
      word.pop_back();
    if (!vocabulary_.emplace(word, index++).second)
      throw std::invalid_argument("Duplicate vocabulary entry");
  }
  for (const auto& token : {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"})
    if (!vocabulary_.contains(token))
      throw std::invalid_argument("Missing special token");
}
static bool cjk(std::int32_t c) {
  return (c >= 0x4e00 && c <= 0x9fff) || (c >= 0x3400 && c <= 0x4dbf) ||
         (c >= 0x20000 && c <= 0x2a6df) || (c >= 0x2a700 && c <= 0x2b73f) ||
         (c >= 0x2b740 && c <= 0x2b81f) || (c >= 0x2b820 && c <= 0x2ceaf) ||
         (c >= 0xf900 && c <= 0xfaff) || (c >= 0x2f800 && c <= 0x2fa1f);
}
std::vector<std::string> WordPieceTokenizer::words(const std::string& text) const {
  utf8proc_uint8_t* normalized = nullptr;
  auto size =
      utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()), text.size(), &normalized,
                   static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_DECOMPOSE));
  if (size < 0)
    throw std::invalid_argument("Invalid UTF-8 text");
  std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> memory(normalized, std::free);
  std::vector<std::string> tokens;
  std::string current;
  auto flush = [&] {
    if (!current.empty()) {
      tokens.push_back(current);
      current.clear();
    }
  };
  for (utf8proc_ssize_t offset = 0; offset < size;) {
    std::int32_t code;
    auto length = utf8proc_iterate(normalized + offset, size - offset, &code);
    if (length < 0)
      throw std::invalid_argument("Invalid normalized UTF-8");
    offset += length;
    auto category = utf8proc_category(code);
    if (code == ' ' || code == '\t' || code == '\n' || code == '\r' ||
        category == UTF8PROC_CATEGORY_ZS || category == UTF8PROC_CATEGORY_ZL ||
        category == UTF8PROC_CATEGORY_ZP) {
      flush();
      continue;
    }
    if (code == 0 || code == 0xfffd || category == UTF8PROC_CATEGORY_CC ||
        category == UTF8PROC_CATEGORY_CF || category == UTF8PROC_CATEGORY_MN)
      continue;
    code = utf8proc_tolower(code);
    utf8proc_uint8_t encoded[4];
    auto count = utf8proc_encode_char(code, encoded);
    std::string value(reinterpret_cast<char*>(encoded), count);
    bool punctuation = (code >= 33 && code <= 47) || (code >= 58 && code <= 64) ||
                       (code >= 91 && code <= 96) || (code >= 123 && code <= 126) ||
                       (category >= UTF8PROC_CATEGORY_PC && category <= UTF8PROC_CATEGORY_PO);
    if (punctuation || cjk(code)) {
      flush();
      tokens.push_back(value);
    } else
      current += value;
  }
  flush();
  return tokens;
}
std::vector<std::int64_t> WordPieceTokenizer::pieces(const std::string& word) const {
  std::vector<std::size_t> boundaries{0};
  for (std::size_t position = 0; position < word.size();) {
    std::int32_t code;
    auto size = utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(word.data() + position),
                                 word.size() - position, &code);
    if (size < 0)
      throw std::invalid_argument("Invalid word UTF-8");
    position += size;
    boundaries.push_back(position);
  }
  if (boundaries.size() > 101)
    return {vocabulary_.at("[UNK]")};
  std::vector<std::int64_t> result;
  std::size_t start = 0;
  while (start + 1 < boundaries.size()) {
    bool found = false;
    for (std::size_t end = boundaries.size() - 1; end > start; --end) {
      std::string piece =
          (start ? "##" : "") + word.substr(boundaries[start], boundaries[end] - boundaries[start]);
      if (auto it = vocabulary_.find(piece); it != vocabulary_.end()) {
        result.push_back(it->second);
        start = end;
        found = true;
        break;
      }
    }
    if (!found)
      return {vocabulary_.at("[UNK]")};
  }
  return result;
}
EncodedText WordPieceTokenizer::encode(const std::string& text, std::size_t maximum) const {
  if (maximum < 2 || maximum > 4096 || text.size() > 100000)
    throw std::invalid_argument("Invalid sequence or text length");
  std::vector<std::int64_t> body;
  for (std::size_t start = 0; start < text.size();) {
    std::size_t nearest = std::string::npos;
    std::string special;
    for (const auto& token : {"[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}) {
      auto position = text.find(token, start);
      if (position < nearest) {
        nearest = position;
        special = token;
      }
    }
    auto end = nearest == std::string::npos ? text.size() : nearest;
    for (const auto& word : words(text.substr(start, end - start))) {
      auto values = pieces(word);
      body.insert(body.end(), values.begin(), values.end());
    }
    if (nearest == std::string::npos)
      break;
    body.push_back(vocabulary_.at(special));
    start = nearest + special.size();
  }
  if (body.size() > maximum - 2)
    body.resize(maximum - 2);
  EncodedText result;
  result.ids.push_back(vocabulary_.at("[CLS]"));
  result.ids.insert(result.ids.end(), body.begin(), body.end());
  result.ids.push_back(vocabulary_.at("[SEP]"));
  result.mask.assign(result.ids.size(), 1);
  result.ids.resize(maximum, vocabulary_.at("[PAD]"));
  result.mask.resize(maximum, 0);
  return result;
}
}  // namespace portfolio
