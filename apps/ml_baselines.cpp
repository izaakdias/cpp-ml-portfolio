#include <limits>

#include "portfolio/cli.hpp"
#include "portfolio/metrics.hpp"
#include "portfolio/registry.hpp"
#include "portfolio/tabular.hpp"
using namespace portfolio;
int main(int argc, char** argv) {
  return guarded([&] {
    Args args(argc, argv);
    args.only({"data", "task", "artifacts", "seed", "help"});
    if (args.has("help")) {
      std::cout << "--data CSV --task binary|regression --artifacts DIR --seed 41\n";
      return;
    }
    bool classification = args.get("task", "binary") == "binary";
    auto path = args.get("data", classification ? "data/wine.csv" : "data/auto-mpg.csv");
    if (!classification && args.get("task") != "regression")
      throw std::invalid_argument("Task must be binary or regression");
    auto split = split_dataset(read_dataset(path), args.integer("seed", 41), classification);
    auto normalizer = Standardizer::fit(split.train.x);
    auto train = normalizer.transform(split.train.x),
         validation = normalizer.transform(split.validation.x),
         test = normalizer.transform(split.test.x);
    Json report = {
        {"source_sha256", sha256(path)},
        {"split_rows",
         {{"train", train.rows()}, {"validation", validation.rows()}, {"test", test.rows()}}}};
    Json model;
    if (classification) {
      double best = -1, threshold = 0.5, selected = 0;
      LogisticRegression classifier;
      for (double l2 : {0.0001, 0.001, 0.01, 0.1}) {
        LogisticRegression candidate;
        candidate.fit(train, split.train.y, 1500, 0.05, l2);
        auto scores = candidate.predict(validation);
        double cut = f1_threshold(split.validation.y, scores);
        double f1 = binary_metrics(split.validation.y, scores, cut).at("f1");
        if (f1 > best) {
          best = f1;
          classifier = std::move(candidate);
          threshold = cut;
          selected = l2;
        }
      }
      report["logistic"] = binary_metrics(split.test.y, classifier.predict(test), threshold);
      report["selected_l2"] = selected;
      report["threshold"] = threshold;
      double best_knn = -1;
      int selected_k = 3;
      KNearestNeighbors neighbors;
      double knn_threshold = 0.5;
      for (int k : {3, 5, 9})
        if (k <= train.rows()) {
          KNearestNeighbors candidate;
          candidate.fit(train, split.train.y, k);
          auto scores = candidate.predict(validation);
          double cut = f1_threshold(split.validation.y, scores);
          double f1 = binary_metrics(split.validation.y, scores, cut).at("f1");
          if (f1 > best_knn) {
            best_knn = f1;
            selected_k = k;
            neighbors = std::move(candidate);
            knn_threshold = cut;
          }
        }
      report["knn_reference"] =
          binary_metrics(split.test.y, neighbors.predict(test), knn_threshold);
      report["knn_selected_k"] = selected_k;
      report["majority_baseline"] =
          binary_metrics(split.test.y, Vector::Constant(test.rows(), split.train.y.mean()), 0.5);
      model = classifier.json();
      model["normalizer"] = normalizer.json();
      model["threshold"] = threshold;
    } else {
      double best = std::numeric_limits<double>::infinity(), selected = 0;
      LinearRegression classifier;
      for (double ridge : {0.00001, 0.01, 1.0}) {
        LinearRegression candidate;
        candidate.fit(train, split.train.y, ridge);
        double mae =
            regression_metrics(split.validation.y, candidate.predict(validation)).at("mae");
        if (mae < best) {
          best = mae;
          classifier = std::move(candidate);
          selected = ridge;
        }
      }
      report["linear"] = regression_metrics(split.test.y, classifier.predict(test));
      report["selected_ridge"] = selected;
      report["mean_baseline"] =
          regression_metrics(split.test.y, Vector::Constant(test.rows(), split.train.y.mean()));
      model = {{"kind", "linear_regression"},
               {"weights", vector_json(classifier.weights)},
               {"normalizer", normalizer.json()}};
    }
    model["features"] = split.train.features;
    auto artifact =
        save_artifact(args.get("artifacts", "artifacts"),
                      classification ? "tabular-classification" : "tabular-regression", model,
                      {{"metrics", report},
                       {"seed", args.integer("seed", 41)},
                       {"policy", "validation_selected_no_refit"}});
    print({{"artifact", artifact.path.string()},
           {"run_id", artifact.metadata["run_id"]},
           {"metrics", report}});
  });
}
