#include "portfolio/cli.hpp"
#include "portfolio/metrics.hpp"
#include "portfolio/registry.hpp"
#include "portfolio/tabular.hpp"
#include "portfolio/tracking.hpp"
using namespace portfolio;
int main(int argc, char** argv) {
  return guarded([&] {
    Args args(argc, argv);
    args.only({"data", "uri", "experiment", "artifacts", "help"});
    if (args.has("help")) {
      std::cout
          << "--data data/auto-mpg.csv --uri http://127.0.0.1:5000 --experiment cpp-portfolio\n";
      return;
    }
    auto split = split_dataset(read_dataset(args.get("data", "data/auto-mpg.csv")));
    auto normalizer = Standardizer::fit(split.train.x);
    LinearRegression model;
    model.fit(normalizer.transform(split.train.x), split.train.y);
    auto metrics =
        regression_metrics(split.test.y, model.predict(normalizer.transform(split.test.x)));
    ExperimentTracker tracking(args.get("uri", "http://127.0.0.1:5000"));
    auto experiment = tracking.experiment(args.get("experiment", "cpp-portfolio"));
    auto run = tracking.start(experiment, "native-linear-regression");
    try {
      tracking.log(run,
                   {{"algorithm", "ridge_qr"},
                    {"seed", 41},
                    {"train_rows", split.train.x.rows()},
                    {"features", split.train.x.cols()}},
                   metrics);
      tracking.finish(run);
    } catch (...) {
      try {
        tracking.finish(run, false);
      } catch (...) {
      }
      throw;
    }
    auto artifact = save_artifact(args.get("artifacts", "artifacts"), "tracked-regression",
                                  {{"kind", "linear_regression"},
                                   {"weights", vector_json(model.weights)},
                                   {"normalizer", normalizer.json()}},
                                  {{"metrics", metrics}, {"mlflow_run_id", run}});
    print({{"mlflow_run_id", run},
           {"experiment_id", experiment},
           {"metrics", metrics},
           {"artifact", artifact.path.string()}});
  });
}
