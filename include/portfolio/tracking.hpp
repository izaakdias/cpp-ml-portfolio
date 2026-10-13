#pragma once
#include <memory>

#include "portfolio/data.hpp"
namespace httplib {
class Client;
}
namespace portfolio {
class ExperimentTracker {
  std::unique_ptr<httplib::Client> client_;
  Json post(const std::string& path, const Json& body);

 public:
  explicit ExperimentTracker(const std::string& uri);
  ~ExperimentTracker();
  std::string experiment(const std::string& name);
  std::string start(const std::string& experiment_id, const std::string& name);
  void log(const std::string& run_id, const Json& parameters, const Json& metrics);
  void finish(const std::string& run_id, bool success = true);
};
}  // namespace portfolio
