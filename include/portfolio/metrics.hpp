#pragma once
#include "portfolio/data.hpp"
namespace portfolio {
Json regression_metrics(const Vector& truth, const Vector& prediction);
Json binary_metrics(const Vector& truth, const Vector& scores, double threshold);
double f1_threshold(const Vector& truth, const Vector& scores);
}  // namespace portfolio
