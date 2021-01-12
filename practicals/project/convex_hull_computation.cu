#include "convex_hull_computation.h"

#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cuda.h>


__global__ void xxx_kernel(const Point* points_array, const int points_count) {

}


bool is_point_on_right_side_of_line(const Point& line_start, const Point& line_end, const Point& point) {
    float point1_x = point.first - line_start.first;
    float point1_y = point.second - line_start.second;
    float point2_x = line_end.first - line_start.first;
    float point2_y = line_end.second - line_start.second;
    return point1_x * point2_y - point1_y * point2_x < 0;
}


float get_squared_distance_of_point_from_line(const Point& line_start, const Point& line_end, const Point& point) {
    float point1_x = point.first - line_start.first;
    float point1_y = point.second - line_start.second;
    float point2_x = line_end.first - line_start.first;
    float point2_y = line_end.second - line_start.second;
    float root_of_enumerator = point1_x * point2_y - point1_y * point2_x;
    float enumerator = root_of_enumerator * root_of_enumerator;
    float denominator = point2_x * point2_x + point2_y * point2_y;
    return enumerator / denominator;
}

Point get_farthest_point_from_line(const Point& line_start, const Point& line_end, const std::vector<Point>& points) {
    if (points.size() == 0) {
        throw std::invalid_argument("A farthest point of empty set of points is not defined");
    }
    Point farthest_point = points[0];
    float max_distance = get_squared_distance_of_point_from_line(line_start, line_end, points[0]);
    for (int i = 1; i < points.size(); i++) {
        float current_distance = get_squared_distance_of_point_from_line(line_start, line_end, points[i]);
        if (current_distance > max_distance) {
            max_distance = current_distance;
            farthest_point = points[i];
        }
    }
    return farthest_point;
}

ConvexHullTime compute_quickhull_gpu_with_time(std::vector<Point>& points, bool include_data_transfer_time) {
    Point* device_points;
    cudaMalloc((void **)&device_points, sizeof(Point) * points.size());
    cudaMemcpy(device_points, &points[0], sizeof(Point) * points.size(), cudaMemcpyHostToDevice);
    float milliseconds_time;
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);
    cudaEventRecord(start_time);
    // TODO
    ConvexHull convex_hull;
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    cudaEventElapsedTime(&milliseconds_time, start_time, end_time);
    return ConvexHullTime(convex_hull, milliseconds_time);
}


void compute_recursive_quickhull_cpu(
    const Point& line_start, const Point& line_end, const std::vector<Point>& points, std::vector<Point>& chosen_points
) {
    if (points.size() == 0) {
        return;
    }
    Point farthest_point = get_farthest_point_from_line(line_start, line_end, points);
    std::vector<Point> side1_points, side2_points;
    for (int i = 0; i < points.size(); i++) {
        if (points[i] == farthest_point) {
            continue;
        }
        if (is_point_on_right_side_of_line(line_start, farthest_point, points[i])) {
            side1_points.push_back(points[i]);
        } else if (is_point_on_right_side_of_line(farthest_point, line_end, points[i])) {
            side2_points.push_back(points[i]);
        }
    }
    compute_recursive_quickhull_cpu(line_start, farthest_point, side1_points, chosen_points);
    chosen_points.push_back(farthest_point);
    compute_recursive_quickhull_cpu(farthest_point, line_end, side2_points, chosen_points);
}

ConvexHull compute_quickhull_cpu(std::vector<Point>& points) {
    std::vector<Point> left_points, right_points, chosen_points;
    std::sort(points.begin(), points.end());
    chosen_points.push_back(points[0]);
    for (int i = 1; i < points.size() - 1; i++) {
        if (is_point_on_right_side_of_line(points[0], points[points.size() - 1], points[i])) {
            right_points.push_back(points[i]);
        } else {
            left_points.push_back(points[i]);
        }
    }
    compute_recursive_quickhull_cpu(
        points[0], points[points.size() - 1], right_points, chosen_points
    );
    chosen_points.push_back(points[points.size() - 1]);
    compute_recursive_quickhull_cpu(
        points[points.size() - 1], points[0], left_points, chosen_points
    );
    chosen_points.push_back(points[0]);
    return ConvexHull(points, chosen_points);
}


ConvexHull compute_graham_cpu(std::vector<Point>& points) {
    return ConvexHull();
}


ConvexHull compute_cpu_convex_hull(const std::string& hull_algorithm, std::vector<Point>& points) {
    if (hull_algorithm == "quickhull_cpu") {
        return compute_quickhull_cpu(points);
    } else if (hull_algorithm == "graham_cpu") {
        return compute_graham_cpu(points);
    } else {
        throw std::invalid_argument("Invalid hull_algorithm: " + hull_algorithm);
    }
}


ConvexHullTime compute_convex_hull_with_time(
    const std::string& hull_algorithm, std::vector<Point>& points, bool include_data_transfer_time
) {
    if (hull_algorithm == "quickhull_gpu") {
        return compute_quickhull_gpu_with_time(points, include_data_transfer_time);
    } else {
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        ConvexHull convex_hull = compute_cpu_convex_hull(hull_algorithm, points);
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        int microseconds_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        float milliseconds_time = microseconds_time / 1000.0;
        return ConvexHullTime(convex_hull, milliseconds_time);
    }
}