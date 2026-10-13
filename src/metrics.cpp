#include "portfolio/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace portfolio {
static void aligned(const Vector& a, const Vector& b) {
  if (!a.size() || a.size() != b.size() || !a.allFinite() || !b.allFinite())
    throw std::invalid_argument("Metrics require aligned finite vectors");
}
Json regression_metrics(const Vector& truth, const Vector& prediction) {
  aligned(truth, prediction);
  auto error = truth - prediction;
  double variance = (truth.array() - truth.mean()).square().sum();
  Json result = {{"mae", error.array().abs().mean()},
                 {"rmse", std::sqrt(error.squaredNorm() / truth.size())}};
  result["r2"] = variance > 0 ? Json(1 - error.squaredNorm() / variance) : Json(nullptr);
  return result;
}
Json binary_metrics(const Vector& truth, const Vector& scores, double threshold) {
  aligned(truth, scores);
  if (!std::isfinite(threshold))
    throw std::invalid_argument("Finite threshold required");
  double tp = 0, fp = 0, fn = 0, tn = 0;
  std::vector<Eigen::Index> order(truth.size());
  std::iota(order.begin(), order.end(), 0);
  for (Eigen::Index i = 0; i < truth.size(); ++i) {
    if (truth(i) != 0 && truth(i) != 1)
      throw std::invalid_argument("Binary labels required");
    if (scores(i) >= threshold) {
      if (truth(i) == 1)
        ++tp;
      else
        ++fp;
    } else {
      if (truth(i) == 1)
        ++fn;
      else
        ++tn;
    }
  }
  double precision = tp + fp ? tp / (tp + fp) : 0;
  double recall = tp + fn ? tp / (tp + fn) : 0;
  Json result = {{"precision", precision},
                 {"recall", recall},
                 {"f1", precision + recall ? 2 * precision * recall / (precision + recall) : 0},
                 {"accuracy", (tp + tn) / truth.size()},
                 {"alert_rate", (tp + fp) / truth.size()}};
  std::stable_sort(order.begin(), order.end(),
                   [&](auto a, auto b) { return scores(a) > scores(b); });
  double positives = tp + fn, negatives = tn + fp, ap = 0, rank_sum = 0, cumulative_positive = 0;
  for (std::size_t start = 0; start < order.size();) {
    std::size_t end = start;
    double group_positive = 0;
    while (end < order.size() && scores(order[end]) == scores(order[start])) {
      group_positive += truth(order[end]);
      ++end;
    }
    cumulative_positive += group_positive;
    if (positives)
      ap += (group_positive / positives) * (cumulative_positive / end);
    double ascending_average_rank = truth.size() - (start + end - 1) / 2.0;
    rank_sum += group_positive * ascending_average_rank;
    start = end;
  }
  result["average_precision"] = positives ? Json(ap) : Json(nullptr);
  result["roc_auc"] =
      positives && negatives
          ? Json((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))
          : Json(nullptr);
  return result;
}
double f1_threshold(const Vector& truth, const Vector& scores) {
  aligned(truth, scores);
  std::vector<double> thresholds(scores.data(), scores.data() + scores.size());
  std::sort(thresholds.begin(), thresholds.end());
  thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());
  double best = -1, result = thresholds.front();
  for (double threshold : thresholds) {
    double f1 = binary_metrics(truth, scores, threshold).at("f1");
    if (f1 >= best) {
      best = f1;
      result = threshold;
    }
  }
  return result;
}
}  // namespace portfolio
