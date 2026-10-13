#include "portfolio/stream.hpp"
namespace portfolio {
DeviceStreams::DeviceStreams(std::size_t window, Eigen::Index features)
    : capacity_(window), features_(features) {
  if (window < 2 || window > 10000 || features < 1 || features > 1000)
    throw std::invalid_argument("Invalid stream dimensions");
}
std::optional<Vector> DeviceStreams::append(const std::string& device, int64_t timestamp,
                                            const Vector& values) {
  if (device.empty() || device.size() > 128 || timestamp < 1 || values.size() != features_ ||
      !values.allFinite())
    throw std::invalid_argument("Invalid device observation");
  if (!devices_.contains(device) && devices_.size() >= 1024)
    throw std::invalid_argument("Device capacity exceeded");
  auto& state = devices_[device];
  if (timestamp <= state.timestamp)
    throw std::invalid_argument("Device timestamps must strictly increase");
  state.timestamp = timestamp;
  state.readings.push_back(values);
  if (state.readings.size() > capacity_)
    state.readings.pop_front();
  RollingWindow window(capacity_, features_);
  std::optional<Vector> summary;
  for (const auto& reading : state.readings) summary = window.append(reading);
  return summary;
}
Json DeviceStreams::json() const {
  Json devices = Json::object();
  for (const auto& [name, state] : devices_) {
    Json rows = Json::array();
    for (const auto& values : state.readings) rows.push_back(vector_json(values));
    devices[name] = {{"timestamp_ms", state.timestamp}, {"readings", rows}};
  }
  return {{"window", capacity_}, {"features", features_}, {"devices", devices}};
}
void DeviceStreams::restore(const Json& json) {
  if (json.at("window") != capacity_ || json.at("features") != features_ ||
      !json.at("devices").is_object() || json.at("devices").size() > 1024)
    throw std::invalid_argument("Invalid stream checkpoint contract");
  DeviceStreams restored(capacity_, features_);
  for (auto it = json.at("devices").begin(); it != json.at("devices").end(); ++it) {
    auto timestamp = it.value().at("timestamp_ms").get<int64_t>();
    auto rows = it.value().at("readings");
    if (!rows.is_array() || rows.empty() || rows.size() > capacity_)
      throw std::invalid_argument("Invalid stream checkpoint window");
    int64_t start = timestamp - static_cast<int64_t>(rows.size()) + 1;
    for (const auto& row : rows) restored.append(it.key(), start++, json_vector(row));
  }
  devices_ = std::move(restored.devices_);
}
}  // namespace portfolio
