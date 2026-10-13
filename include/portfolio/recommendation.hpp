#pragma once
#include <map>
#include <set>

#include "portfolio/data.hpp"
namespace portfolio {
struct Rating {
  std::string user, item;
  double value;
  std::int64_t timestamp;
};
std::vector<Rating> read_ratings(const std::filesystem::path& path);
std::pair<std::vector<Rating>, std::vector<Rating>> holdout_latest(
    const std::vector<Rating>& ratings);
class MatrixFactorization {
  Matrix users_, items_;
  Vector user_bias_, item_bias_;
  double mean_ = 0;
  std::map<std::string, int> user_index_, item_index_;
  std::map<std::string, std::set<std::string>> seen_;

 public:
  void fit(const std::vector<Rating>& ratings, int dimensions = 16, int epochs = 20,
           unsigned seed = 61);
  double predict(const std::string& user, const std::string& item) const;
  Json recommend(const std::string& user, std::size_t limit = 10) const;
  Json json() const;
  static MatrixFactorization from_json(const Json& json);
};
}  // namespace portfolio
