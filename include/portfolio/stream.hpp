#pragma once
#include <map>

#include "portfolio/isolation_forest.hpp"
namespace portfolio {
class DeviceStreams {
  struct Device {
    int64_t timestamp = 0;
    std::deque<Vector> readings;
  };
  std::map<std::string, Device> devices_;
  std::size_t capacity_;
  Eigen::Index features_;

 public:
  DeviceStreams(std::size_t window, Eigen::Index features);
  std::optional<Vector> append(const std::string& device, int64_t timestamp, const Vector& values);
  Json json() const;
  void restore(const Json& state);
};
}  // namespace portfolio
