#include <httplib.h>

#include <catch2/catch_test_macros.hpp>
#include <thread>

#include "portfolio/tracking.hpp"
using namespace portfolio;
TEST_CASE("Tracking client creates an experiment logs a run and finishes") {
  httplib::Server server;
  Json recorded;
  server.Get("/api/2.0/mlflow/experiments/get-by-name", [](const auto&, auto& response) {
    response.status = 404;
    response.set_content("{}", "application/json");
  });
  server.Post("/api/2.0/mlflow/experiments/create", [](const auto&, auto& response) {
    response.set_content("{\"experiment_id\":\"1\"}", "application/json");
  });
  server.Post("/api/2.0/mlflow/runs/create", [](const auto&, auto& response) {
    response.set_content("{\"run\":{\"info\":{\"run_id\":\"run-1\"}}}", "application/json");
  });
  server.Post("/api/2.0/mlflow/runs/log-batch", [&](const auto& request, auto& response) {
    recorded = Json::parse(request.body);
    response.set_content("{}", "application/json");
  });
  server.Post("/api/2.0/mlflow/runs/update",
              [](const auto&, auto& response) { response.set_content("{}", "application/json"); });
  auto port = server.bind_to_any_port("127.0.0.1");
  REQUIRE(port > 0);
  std::thread thread([&] { server.listen_after_bind(); });
  struct Shutdown {
    httplib::Server& server;
    std::thread& thread;
    ~Shutdown() {
      server.stop();
      thread.join();
    }
  } shutdown{server, thread};
  ExperimentTracker client("http://127.0.0.1:" + std::to_string(port));
  auto experiment = client.experiment("tests");
  auto run = client.start(experiment, "native");
  client.log(run, {{"seed", 41}}, {{"rmse", 0.1}});
  client.finish(run);
  REQUIRE(recorded["run_id"] == "run-1");
  REQUIRE(recorded["metrics"][0]["value"] == 0.1);
}
