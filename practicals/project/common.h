#ifndef CUDA_PRAC1_COMMON_H
#define CUDA_PRAC1_COMMON_H

#include <vector>
#include <utility>

typedef std::pair<float, float> Point;
typedef std::pair<std::vector<Point>, std::vector<Point>> ConvexHull;
typedef std::pair<ConvexHull, float> ConvexHullTime;

#endif //CUDA_PRAC1_COMMON_H
