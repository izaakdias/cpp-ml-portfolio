#include <httplib.h>

#include <semaphore>

#include "portfolio/cli.hpp"
#include "portfolio/inference.hpp"
#include "portfolio/registry.hpp"
using namespace portfolio;
int main(int argc, char** argv) {
  return guarded([&] {
    Args args(argc, argv);
    args.only({"model", "input", "mode", "threads", "provider", "warmup", "runs", "host", "port",
               "workers", "help"});
    if (args.has("help")) {
      std::cout << "--model ONNX --mode inspect|infer|benchmark|serve --input JSON --provider "
                   "CPU|CoreML\n";
      return;
    }
    InferenceEngine engine(args.get("model", "models/linear.onnx"), args.integer("threads", 1),
                           args.get("provider", "CPU"));
    auto mode = args.get("mode", "inspect");
    if (mode == "inspect")
      print(engine.describe());
    else if (mode == "infer")
      print(engine.run(read_json(args.get("input", "models/linear-input.json")).at("inputs")));
    else if (mode == "benchmark")
      print(benchmark(engine, read_json(args.get("input", "models/linear-input.json")).at("inputs"),
                      args.integer("warmup", 10), args.integer("runs", 100)));
    else if (mode == "serve") {
      int workers = args.integer("workers", 4), port = args.integer("port", 8080);
      if (workers < 1 || workers > 32 || port < 1 || port > 65535)
        throw std::invalid_argument("Invalid worker count or port");
      httplib::Server server;
      server.new_task_queue = [workers] { return new httplib::ThreadPool(workers, workers, 64); };
      server.set_payload_max_length(4 * 1024 * 1024);
      std::counting_semaphore<64> slots(workers);
      server.Get("/health", [&](const auto&, auto& response) {
        response.set_content(engine.describe().dump(), "application/json");
      });
      server.Post("/infer", [&](const auto& request, auto& response) {
        if (!slots.try_acquire()) {
          response.status = 503;
          response.set_content("{\"error\":\"capacity_exceeded\"}", "application/json");
          return;
        }
        struct Release {
          std::counting_semaphore<64>& slots;
          ~Release() {
            slots.release();
          }
        } release{slots};
        try {
          auto body = Json::parse(request.body);
          if (body.size() != 1 || !body.contains("inputs"))
            throw std::invalid_argument("Request requires exactly inputs");
          response.set_content(engine.run(body.at("inputs")).dump(), "application/json");
        } catch (const Json::exception&) {
          response.status = 400;
          response.set_content("{\"error\":\"invalid_json_contract\"}", "application/json");
        } catch (const std::invalid_argument& error) {
          response.status = 400;
          response.set_content(Json({{"error", error.what()}}).dump(), "application/json");
        } catch (const std::exception&) {
          response.status = 500;
          response.set_content("{\"error\":\"inference_failed\"}", "application/json");
        }
      });
      std::cerr << "Listening on " << args.get("host", "127.0.0.1") << ':' << port << '\n';
      if (!server.listen(args.get("host", "127.0.0.1"), port))
        throw std::runtime_error("Cannot listen");
    } else
      throw std::invalid_argument("Unknown mode");
  });
}
