#pragma once
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>

#include "portfolio/data.hpp"
namespace portfolio {
class Args {
  std::map<std::string, std::string> values_;

 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string key = argv[i];
      if (key.rfind("--", 0) != 0 || key.size() < 3)
        throw std::invalid_argument("Use named --arguments");
      key = key.substr(2);
      if (values_.contains(key))
        throw std::invalid_argument("Duplicate argument: " + key);
      values_[key] =
          i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0 ? argv[++i] : "true";
    }
  }
  std::string get(const std::string& name, const std::string& fallback = "") const {
    auto it = values_.find(name);
    return it == values_.end() ? fallback : it->second;
  }
  int integer(const std::string& name, int fallback) const {
    auto value = get(name, std::to_string(fallback));
    std::size_t end;
    auto result = std::stoi(value, &end);
    if (end != value.size())
      throw std::invalid_argument("Integer required: " + name);
    return result;
  }
  bool has(const std::string& name) const {
    return values_.contains(name);
  }
  void only(std::initializer_list<std::string> allowed) const {
    for (const auto& [key, value] : values_) {
      (void)value;
      if (std::find(allowed.begin(), allowed.end(), key) == allowed.end())
        throw std::invalid_argument("Unknown argument: " + key);
    }
  }
};
inline int guarded(const std::function<void()>& run) {
  try {
    run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << Json({{"error", error.what()}}).dump() << '\n';
    return 2;
  }
}
inline void print(const Json& json) {
  std::cout << json.dump(2) << '\n';
}
}  // namespace portfolio
