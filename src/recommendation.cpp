#include "portfolio/recommendation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
namespace portfolio {
std::vector<Rating> read_ratings(const std::filesystem::path& path) {
  auto rows = read_csv(path);
  if (rows.front() != std::vector<std::string>{"user_id", "item_id", "rating", "timestamp"})
    throw std::invalid_argument("Expected ratings CSV header user_id,item_id,rating,timestamp");
  std::vector<Rating> ratings;
  std::set<std::pair<std::string, std::string>> keys;
  for (std::size_t i = 1; i < rows.size(); ++i) {
    double value = number(rows[i][2]);
    double timestamp = number(rows[i][3]);
    if (rows[i][0].empty() || rows[i][1].empty() || value < 1 || value > 5 || timestamp < 0 ||
        timestamp > 9e15 || std::floor(timestamp) != timestamp ||
        !keys.insert({rows[i][0], rows[i][1]}).second)
      throw std::invalid_argument("Invalid or duplicate rating");
    ratings.push_back({rows[i][0], rows[i][1], value, static_cast<std::int64_t>(timestamp)});
  }
  return ratings;
}
std::pair<std::vector<Rating>, std::vector<Rating>> holdout_latest(
    const std::vector<Rating>& ratings) {
  std::map<std::string, std::vector<std::size_t>> groups;
  for (std::size_t i = 0; i < ratings.size(); ++i) groups[ratings[i].user].push_back(i);
  std::set<std::size_t> heldout;
  for (const auto& [user, indices] : groups) {
    (void)user;
    if (indices.size() < 2)
      throw std::invalid_argument("Two interactions per user required");
    heldout.insert(*std::max_element(indices.begin(), indices.end(), [&](auto a, auto b) {
      return std::pair(ratings[a].timestamp, a) < std::pair(ratings[b].timestamp, b);
    }));
  }
  std::vector<Rating> train, test;
  for (std::size_t i = 0; i < ratings.size(); ++i)
    (heldout.contains(i) ? test : train).push_back(ratings[i]);
  return {train, test};
}
void MatrixFactorization::fit(const std::vector<Rating>& ratings, int dimensions, int epochs,
                              unsigned seed) {
  if (ratings.empty() || dimensions < 1 || dimensions > 256 || epochs < 1)
    throw std::invalid_argument("Invalid factorization parameters");
  user_index_.clear();
  item_index_.clear();
  seen_.clear();
  mean_ = 0;
  for (const auto& rating : ratings) {
    if (!std::isfinite(rating.value) || rating.value < 1 || rating.value > 5)
      throw std::invalid_argument("Invalid rating range");
    user_index_.try_emplace(rating.user, 0);
    item_index_.try_emplace(rating.item, 0);
    seen_[rating.user].insert(rating.item);
    mean_ += rating.value;
  }
  mean_ /= ratings.size();
  int index = 0;
  for (auto& [id, i] : user_index_) {
    (void)id;
    i = index++;
  }
  index = 0;
  for (auto& [id, i] : item_index_) {
    (void)id;
    i = index++;
  }
  users_ = Matrix(user_index_.size(), dimensions);
  items_ = Matrix(item_index_.size(), dimensions);
  user_bias_ = Vector::Zero(users_.rows());
  item_bias_ = Vector::Zero(items_.rows());
  std::mt19937 rng(seed);
  std::normal_distribution<double> normal(0, 0.1);
  for (auto& value : users_.reshaped()) value = normal(rng);
  for (auto& value : items_.reshaped()) value = normal(rng);
  std::vector<std::size_t> order(ratings.size());
  std::iota(order.begin(), order.end(), 0);
  constexpr double rate = 0.01, regularization = 0.03;
  for (int epoch = 0; epoch < epochs; ++epoch) {
    std::shuffle(order.begin(), order.end(), rng);
    for (auto row : order) {
      const auto& rating = ratings[row];
      int user = user_index_.at(rating.user), item = item_index_.at(rating.item);
      double error = rating.value - (mean_ + user_bias_(user) + item_bias_(item) +
                                     users_.row(user).dot(items_.row(item)));
      Vector previous = users_.row(user).transpose();
      users_.row(user) += rate * (error * items_.row(item) - regularization * users_.row(user));
      items_.row(item) += rate * (error * previous.transpose() - regularization * items_.row(item));
      user_bias_(user) += rate * (error - regularization * user_bias_(user));
      item_bias_(item) += rate * (error - regularization * item_bias_(item));
    }
    if (!users_.allFinite() || !items_.allFinite())
      throw std::runtime_error("Factorization diverged");
  }
}
double MatrixFactorization::predict(const std::string& user, const std::string& item) const {
  if (!user_index_.contains(user))
    throw std::out_of_range("Unknown user ID");
  int u = user_index_.at(user);
  if (!item_index_.contains(item))
    return std::clamp(mean_ + user_bias_(u), 1.0, 5.0);
  int i = item_index_.at(item);
  return std::clamp(mean_ + user_bias_(u) + item_bias_(i) + users_.row(u).dot(items_.row(i)), 1.0,
                    5.0);
}
Json MatrixFactorization::recommend(const std::string& user, std::size_t limit) const {
  if (!user_index_.contains(user))
    throw std::out_of_range("Unknown user ID");
  if (limit < 1 || limit > 100)
    throw std::invalid_argument("Recommendation limit must be 1..100");
  std::vector<std::pair<double, std::string>> scores;
  for (const auto& [id, index] : item_index_) {
    (void)index;
    if (!seen_.at(user).contains(id))
      scores.emplace_back(predict(user, id), id);
  }
  std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) {
    return a.first == b.first ? a.second < b.second : a.first > b.first;
  });
  Json result = Json::array();
  for (std::size_t i = 0; i < std::min(limit, scores.size()); ++i)
    result.push_back({{"item_id", scores[i].second}, {"predicted_rating", scores[i].first}});
  return result;
}
Json MatrixFactorization::json() const {
  Json users = Json::object(), items = Json::object();
  for (const auto& [id, index] : user_index_)
    users[id] = {{"bias", user_bias_(index)},
                 {"factors", vector_json(users_.row(index).transpose())},
                 {"seen", seen_.at(id)}};
  for (const auto& [id, index] : item_index_)
    items[id] = {{"bias", item_bias_(index)},
                 {"factors", vector_json(items_.row(index).transpose())}};
  return {{"kind", "explicit_rating_matrix_factorization"},
          {"mean", mean_},
          {"users", users},
          {"items", items}};
}
MatrixFactorization MatrixFactorization::from_json(const Json& json) {
  if (json.at("kind") != "explicit_rating_matrix_factorization")
    throw std::invalid_argument("Invalid recommender kind");
  const auto& users = json.at("users");
  const auto& items = json.at("items");
  if (!users.is_object() || !items.is_object() || users.empty() || items.empty())
    throw std::invalid_argument("Empty factorization snapshot");
  MatrixFactorization model;
  model.mean_ = json.at("mean");
  if (!std::isfinite(model.mean_) || model.mean_ < 1 || model.mean_ > 5)
    throw std::invalid_argument("Invalid rating mean");
  auto dimensions = users.begin().value().at("factors").size();
  if (dimensions < 1 || dimensions > 256)
    throw std::invalid_argument("Invalid latent dimension");
  model.users_ = Matrix(users.size(), dimensions);
  model.items_ = Matrix(items.size(), dimensions);
  model.user_bias_ = Vector(users.size());
  model.item_bias_ = Vector(items.size());
  int index = 0;
  for (auto it = items.begin(); it != items.end(); ++it, ++index) {
    auto factors = json_vector(it.value().at("factors"));
    double bias = it.value().at("bias");
    if (factors.size() != static_cast<Eigen::Index>(dimensions) || !std::isfinite(bias))
      throw std::invalid_argument("Invalid item factors");
    model.item_index_[it.key()] = index;
    model.items_.row(index) = factors.transpose();
    model.item_bias_(index) = bias;
  }
  index = 0;
  for (auto it = users.begin(); it != users.end(); ++it, ++index) {
    auto factors = json_vector(it.value().at("factors"));
    double bias = it.value().at("bias");
    if (factors.size() != static_cast<Eigen::Index>(dimensions) || !std::isfinite(bias))
      throw std::invalid_argument("Invalid user factors");
    model.user_index_[it.key()] = index;
    model.users_.row(index) = factors.transpose();
    model.user_bias_(index) = bias;
    model.seen_[it.key()] = it.value().at("seen").get<std::set<std::string>>();
    for (const auto& item : model.seen_.at(it.key()))
      if (!model.item_index_.contains(item))
        throw std::invalid_argument("Unknown seen item in snapshot");
  }
  return model;
}
}  // namespace portfolio
