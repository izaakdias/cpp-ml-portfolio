#include "portfolio/tracking.hpp"

#include <httplib.h>

#include <chrono>
#include <cstdlib>
#include <stdexcept>
namespace portfolio {
ExperimentTracker::ExperimentTracker(const std::string& uri)
    : client_(std::make_unique<httplib::Client>(uri)) {
  if (uri.rfind("http://", 0) != 0 && uri.rfind("https://", 0) != 0)
    throw std::invalid_argument("MLflow requires an HTTP(S) URI");
  client_->set_connection_timeout(5);
  client_->set_read_timeout(10);
  client_->set_write_timeout(10);
  if (const char* token = std::getenv("MLFLOW_TRACKING_TOKEN"))
    client_->set_default_headers({{"Authorization", std::string("Bearer ") + token}});
}
ExperimentTracker::~ExperimentTracker() = default;
Json ExperimentTracker::post(const std::string& path, const Json& body) {
  auto result = client_->Post(path, body.dump(), "application/json");
  if (!result)
    throw std::runtime_error("MLflow connection failed");
  if (result->status != 200)
    throw std::runtime_error("MLflow request failed with HTTP " + std::to_string(result->status));
  return Json::parse(result->body);
}
std::string ExperimentTracker::experiment(const std::string& name) {
  if (name.empty())
    throw std::invalid_argument("Experiment name cannot be empty");
  auto response = client_->Get("/api/2.0/mlflow/experiments/get-by-name",
                               httplib::Params{{"experiment_name", name}});
  if (!response)
    throw std::runtime_error("MLflow connection failed");
  if (response->status == 200)
    return Json::parse(response->body).at("experiment").at("experiment_id");
  if (response->status != 404)
    throw std::runtime_error("MLflow experiment lookup failed with HTTP " +
                             std::to_string(response->status));
  return post("/api/2.0/mlflow/experiments/create", {{"name", name}}).at("experiment_id");
}
std::string ExperimentTracker::start(const std::string& experiment_id, const std::string& name) {
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  return post("/api/2.0/mlflow/runs/create",
              {{"experiment_id", experiment_id},
               {"start_time", timestamp},
               {"tags", Json::array({{{"key", "mlflow.runName"}, {"value", name}}})}})
      .at("run")
      .at("info")
      .at("run_id");
}
void ExperimentTracker::log(const std::string& run_id, const Json& parameters,
                            const Json& metrics) {
  Json params = Json::array(), values = Json::array();
  for (auto it = parameters.begin(); it != parameters.end(); ++it)
    params.push_back(
        {{"key", it.key()},
         {"value", it.value().is_string() ? it.value().get<std::string>() : it.value().dump()}});
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
  for (auto it = metrics.begin(); it != metrics.end(); ++it)
    if (it.value().is_number()) {
      double value = it.value();
      if (!std::isfinite(value))
        throw std::invalid_argument("Finite tracking metrics required");
      values.push_back(
          {{"key", it.key()}, {"value", value}, {"timestamp", timestamp}, {"step", 0}});
    }
  post("/api/2.0/mlflow/runs/log-batch",
       {{"run_id", run_id}, {"params", params}, {"metrics", values}, {"tags", Json::array()}});
}
void ExperimentTracker::finish(const std::string& run_id, bool success) {
  auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::system_clock::now().time_since_epoch()).count();
  post("/api/2.0/mlflow/runs/update",
       {{"run_id", run_id}, {"status", success ? "FINISHED" : "FAILED"}, {"end_time",end}});
}
}  // namespace portfolio
