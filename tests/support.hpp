#pragma once
#include <chrono>
#include <fstream>
#include <random>

#include "portfolio/data.hpp"
struct TempDirectory {
  std::filesystem::path path;
  TempDirectory() {
    std::random_device random;
    path = std::filesystem::temp_directory_path() /
           ("cpp-portfolio-" + std::to_string(random()) + '-' + std::to_string(random()));
    std::filesystem::create_directory(path);
  }
  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
  std::filesystem::path file(const std::string& name, const std::string& content) {
    auto target = path / name;
    std::ofstream output(target);
    output << content;
    return target;
  }
};
