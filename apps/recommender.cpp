#include "portfolio/cli.hpp"
#include "portfolio/metrics.hpp"
#include "portfolio/recommendation.hpp"
#include "portfolio/registry.hpp"
using namespace portfolio;
int main(int argc, char** argv) {
  return guarded([&] {
    Args args(argc, argv);
    args.only({"data", "user", "limit", "epochs", "artifacts", "mode", "help"});
    if (args.has("help")) {
      std::cout << "--data data/movielens.csv --user 1 --limit 10 --epochs 20\n";
      return;
    }
    auto mode = args.get("mode", "train");
    if (mode == "recommend") {
      auto artifact = load_artifact(args.get("artifacts", "artifacts"), "recommendation");
      auto model = MatrixFactorization::from_json(artifact.model);
      print(
          {{"run_id", artifact.metadata.at("run_id")},
           {"recommendations", model.recommend(args.get("user", "1"), args.integer("limit", 10))}});
      return;
    }
    if (mode != "train")
      throw std::invalid_argument("Mode must be train or recommend");
    auto path = args.get("data", "data/movielens.csv");
    auto [train, test] = holdout_latest(read_ratings(path));
    MatrixFactorization model;
    model.fit(train, 16, args.integer("epochs", 20));
    std::map<std::string, int> popularity;
    std::map<std::string, std::set<std::string>> seen;
    double mean = 0;
    for (const auto& rating : train) {
      ++popularity[rating.item];
      seen[rating.user].insert(rating.item);
      mean += rating.value;
    }
    mean /= train.size();
    std::vector<std::pair<int, std::string>> popular;
    for (const auto& [id, count] : popularity) popular.emplace_back(count, id);
    std::sort(popular.begin(), popular.end(), [](const auto& a, const auto& b) {
      return a.first == b.first ? a.second < b.second : a.first > b.first;
    });
    std::size_t baseline_hits = 0;
    Vector truth(test.size()), predictions(test.size());
    std::size_t hits = 0, relevant = 0;
    double discounted = 0;
    for (std::size_t i = 0; i < test.size(); ++i) {
      truth(i) = test[i].value;
      predictions(i) = model.predict(test[i].user, test[i].item);
      if (test[i].value >= 4) {
        ++relevant;
        int count = 0;
        for (const auto& [frequency, id] : popular) {
          (void)frequency;
          if (seen.at(test[i].user).contains(id))
            continue;
          if (id == test[i].item)
            ++baseline_hits;
          if (++count == 10)
            break;
        }
        auto items = model.recommend(test[i].user, 10);
        for (std::size_t rank = 0; rank < items.size(); ++rank)
          if (items[rank]["item_id"] == test[i].item) {
            ++hits;
            discounted += 1 / std::log2(rank + 2);
            break;
          }
      }
    }
    Json metrics = regression_metrics(truth, predictions);
    metrics["hit_rate_at_10_positive_holdouts"] =
        relevant ? Json(static_cast<double>(hits) / relevant) : Json(nullptr);
    metrics["ndcg_at_10_positive_holdouts"] =
        relevant ? Json(discounted / relevant) : Json(nullptr);
    metrics["positive_holdouts"] = relevant;
    metrics["popularity_baseline_hit_rate_at_10"] =
        relevant ? Json(double(baseline_hits) / relevant) : Json(nullptr);
    metrics["mean_rating_baseline"] =
        regression_metrics(truth, Vector::Constant(truth.size(), mean));
    metrics["training_ratings"] = train.size();
    metrics["test_ratings"] = test.size();
    auto artifact = save_artifact(
        args.get("artifacts", "artifacts"), "recommendation", model.json(),
        {{"metrics", metrics}, {"holdout", "latest_per_user"}, {"source_sha256", sha256(path)}});
    print({{"metrics", metrics},
           {"recommendations", model.recommend(args.get("user", "1"), args.integer("limit", 10))},
           {"artifact", artifact.path.string()}});
  });
}
