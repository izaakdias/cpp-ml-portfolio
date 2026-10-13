#include <algorithm>
#include <fstream>

#include "portfolio/cli.hpp"
#include "portfolio/isolation_forest.hpp"
#include "portfolio/metrics.hpp"
#include "portfolio/registry.hpp"
#include "portfolio/stream.hpp"
using namespace portfolio;
static Dataset slice(const Dataset& raw, Eigen::Index start, Eigen::Index count) {
  return {raw.x.middleRows(start, count), raw.y.segment(start, count), raw.features};
}
static Matrix normals(const Dataset& data) {
  Eigen::Index count = (data.y.array() == 0).count();
  Matrix x(count, data.x.cols());
  Eigen::Index next = 0;
  for (Eigen::Index i = 0; i < data.y.size(); ++i)
    if (data.y(i) == 0)
      x.row(next++) = data.x.row(i);
  return x;
}
int main(int argc, char** argv) {
  return guarded([&] {
    Args args(argc, argv);
    args.only({"data", "mode", "artifacts", "stream", "state", "help"});
    if (args.has("help")) {
      std::cout << "--data data/nab.csv --mode train|replay --artifacts artifacts\n";
      return;
    }
    auto path = args.get("data", "data/nab.csv"), root = args.get("artifacts", "artifacts");
    if (args.get("mode") == "stream") {
      auto state_root = args.get("state", "artifacts/stream-state");
      std::optional<Artifact> saved;
      if (std::filesystem::exists(std::filesystem::path(state_root) / "anomaly-stream/latest.json"))
        saved = load_artifact(state_root, "anomaly-stream");
      auto artifact = load_artifact(
          root, "anomaly", saved ? saved->metadata.at("model_run_id").get<std::string>() : "");
      auto model = IsolationForest::from_json(artifact.model);
      DeviceStreams streams(artifact.metadata.at("window"),
                            artifact.metadata.at("raw_features").size());
      if (saved)
        streams.restore(saved->model);
      auto reference = Standardizer::from_json(artifact.metadata.at("reference"));
      double threshold = artifact.metadata.at("threshold");
      std::ifstream input(args.get("stream", "data/nab-stream.jsonl"));
      if (!input)
        throw std::runtime_error("Cannot open observation stream");
      std::string line;
      int count = 0;
      while (std::getline(input, line)) {
        auto observation = Json::parse(line);
        auto summary = streams.append(observation.at("device"), observation.at("timestamp_ms"),
                                      json_vector(observation.at("values")));
        Json event = {{"device", observation.at("device")},
                      {"timestamp_ms", observation.at("timestamp_ms")},
                      {"model_run_id", artifact.metadata.at("run_id")},
                      {"warming_up", !summary.has_value()}};
        if (summary) {
          Matrix row(1, summary->size());
          row.row(0) = summary->transpose();
          double score = model.predict(row)(0);
          double drift = reference.transform(row).array().abs().mean();
          event["score"] = score;
          event["alert"] = score >= threshold;
          event["reference_distance"] = drift;
          event["distribution_warning"] = drift > 3;
        }
        std::cout << event.dump() << '\n';
        ++count;
      }
      if (!input.eof())
        throw std::runtime_error("Stream read failed");
      auto checkpoint = save_artifact(
          state_root, "anomaly-stream", streams.json(),
          {{"model_run_id", artifact.metadata.at("run_id")}, {"observations_in_batch", count}});
      std::cerr << Json({{"checkpoint", checkpoint.path.string()}}).dump() << '\n';
      return;
    }
    auto raw = read_dataset(path);
    if (args.get("mode", "train") == "train") {
      auto a = raw.x.rows() * 6 / 10, b = raw.x.rows() * 8 / 10;
      auto train = window_dataset(slice(raw, 0, a)),
           validation = window_dataset(slice(raw, a, b - a)),
           test = window_dataset(slice(raw, b, raw.x.rows() - b));
      auto healthy = normals(train);
      IsolationForest model;
      model.fit(healthy);
      double threshold;
      std::string policy;
      if (validation.y.minCoeff() == 0 && validation.y.maxCoeff() == 1) {
        threshold = f1_threshold(validation.y, model.predict(validation.x));
        policy = "validation_max_f1";
      } else {
        auto normal = normals(validation);
        if (normal.rows() == 0)
          throw std::invalid_argument("Validation needs healthy observations");
        auto scores = model.predict(normal);
        std::vector<double> ordered(scores.data(), scores.data() + scores.size());
        std::sort(ordered.begin(), ordered.end());
        threshold = ordered[static_cast<std::size_t>(std::ceil(0.98 * ordered.size())) - 1];
        policy = "validation_healthy_98th_percentile";
      }
      auto metrics = binary_metrics(test.y, model.predict(test.x), threshold);
      metrics["test_windows"] = test.x.rows();
      metrics["threshold_policy"] = policy;
      metrics["validation"] = binary_metrics(validation.y, model.predict(validation.x), threshold);
      auto reference = Standardizer::fit(healthy);
      auto artifact = save_artifact(root, "anomaly", model.json(),
                                    {{"metrics", metrics},
                                     {"threshold", threshold},
                                     {"reference", reference.json()},
                                     {"features", train.features},
                                     {"window", 24},
                                     {"raw_features", raw.features},
                                     {"source_sha256", sha256(path)}});
      print({{"metrics", metrics}, {"artifact", artifact.path.string()}});
    } else if (args.get("mode") == "replay") {
      auto artifact = load_artifact(root, "anomaly");
      auto model = IsolationForest::from_json(artifact.model);
      auto data = window_dataset(raw, artifact.metadata.at("window"));
      auto scores = model.predict(data.x);
      double threshold = artifact.metadata.at("threshold");
      print({{"run_id", artifact.metadata.at("run_id")},
             {"scored_windows", scores.size()},
             {"alerts", (scores.array() >= threshold).count()},
             {"note", "Replay of the supplied series; replay may include training observations."}});
    } else
      throw std::invalid_argument("Mode must be train or replay");
  });
}
