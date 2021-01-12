#ifndef CUDA_PRAC1_CONVEX_HULL_COMPUTATION_H
#define CUDA_PRAC1_CONVEX_HULL_COMPUTATION_H

#include <utility>
#include <vector>
#include <string>
#include "common.h"

ConvexHullTime compute_quickhull_gpu_with_time(std::vector<Point>& points, bool include_data_transfer_time);
ConvexHull compute_quickhull_cpu(std::vector<Point>& points);
ConvexHull compute_graham_cpu(std::vector<Point>& points);
ConvexHull compute_cpu_convex_hull(const std::string& hull_algorithm, std::vector<Point>& points);
ConvexHullTime compute_convex_hull_with_time(
    const std::string& hull_algorithm, std::vector<Point>& points, bool include_data_transfer_time
);

#endif //CUDA_PRAC1_CONVEX_HULL_COMPUTATION_H
