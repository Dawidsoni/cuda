#include "convex_hull_computation.h"

#include <string>
#include <math.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cuda.h>
#include <limits>


__device__ int lock = 0;
__device__ __managed__ int initial_points_count = 0;
__device__ __managed__ int points_count = 0;
__device__ __managed__ int next_points_count = 0;
__device__ __managed__ int segments_count = 0;
__device__ __managed__ int next_segments_count = 0;
__device__ __managed__ bool hull_changed = false;


__device__ float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed, __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}


struct DeviceData {
private:
    std::vector<Point> &h_points;
    Point* h_segments;
    int segments_count;

public:
    Point* d_initial_points;
    Point* d_side_points;
    Point* d_next_side_points;
    Point* d_initial_segments;
    Point* d_side_segments;
    Point* d_next_side_segments;
    int* d_labels;
    int* d_next_labels;
    int* d_new_segment_indexes;
    int* d_segments_count_prefix_sums;
    bool* d_is_on_left_side;
    bool* d_is_on_right_side;
    float* d_segment_max_dists;
    float* d_line_distances;

    DeviceData(
        std::vector<Point> &h_points_, Point *h_segments_, int segments_count_ = 2
    ) : h_points(h_points_), h_segments(h_segments_), segments_count(segments_count_) {}

    void init_memory() {
        int points_count = h_points.size();
        cudaMalloc((void **) &d_initial_points, sizeof(Point) * points_count);
        cudaMalloc((void **) &d_side_points, sizeof(Point) * points_count);
        cudaMalloc((void **) &d_next_side_points, sizeof(Point) * points_count);
        cudaMalloc((void **) &d_initial_segments, sizeof(Point) * points_count);
        cudaMalloc((void **) &d_side_segments, sizeof(Point) * points_count);
        cudaMalloc((void **) &d_next_side_segments, sizeof(Point) * points_count);
        cudaMalloc((void **) &d_labels, sizeof(int) * points_count);
        cudaMalloc((void **) &d_next_labels, sizeof(int) * points_count);
        cudaMalloc((void **) &d_new_segment_indexes, sizeof(int) * points_count);
        cudaMalloc((void **) &d_segments_count_prefix_sums, sizeof(int) * points_count);
        cudaMalloc((void **) &d_is_on_left_side, sizeof(bool) * points_count);
        cudaMalloc((void **) &d_is_on_right_side, sizeof(bool) * points_count);
        cudaMalloc((void **) &d_segment_max_dists, sizeof(float) * points_count);
        cudaMalloc((void **) &d_line_distances, sizeof(float) * points_count);
        cudaMemcpy(d_initial_points, &h_points[0], sizeof(Point) * points_count, cudaMemcpyHostToDevice);
        cudaMemcpy(d_initial_segments, h_segments, sizeof(Point) * segments_count, cudaMemcpyHostToDevice);
    }

    void free_memory() {
        cudaFree(d_initial_points);
        cudaFree(d_side_points);
        cudaFree(d_next_side_points);
        cudaFree(d_initial_segments);
        cudaFree(d_side_segments);
        cudaFree(d_next_side_segments);
        cudaFree(d_labels);
        cudaFree(d_next_labels);
        cudaFree(d_new_segment_indexes);
        cudaFree(d_segments_count_prefix_sums);
        cudaFree(d_is_on_left_side);
        cudaFree(d_is_on_right_side);
        cudaFree(d_segment_max_dists);
        cudaFree(d_line_distances);
    }

    void reset_variables() {
        cudaMemset(d_labels, 0, sizeof(int) * h_points.size());
    }

    void swap_side_points_pointers() {
        Point* d_tmp_pointer;
        d_tmp_pointer = d_side_points;
        d_side_points = d_next_side_points;
        d_next_side_points = d_tmp_pointer;
    }

    void swap_side_segments_pointers() {
        Point* d_tmp_pointer;
        d_tmp_pointer = d_side_segments;
        d_side_segments = d_next_side_segments;
        d_next_side_segments = d_tmp_pointer;
    }

    void swap_labels_pointers() {
        int* tmp_pointer;
        tmp_pointer = d_labels;
        d_labels = d_next_labels;
        d_next_labels = tmp_pointer;
    }
};


struct KernelOptions {
    int threads;
    int blocks;
    int shared_mem;

    KernelOptions(int points_count, int threads_count) {
        this->threads = threads_count;
        this->shared_mem = 2 * threads_count * sizeof(Point);
        update_points_count(points_count);
    }

    void update_points_count(int points_count) {
        this->blocks = static_cast<int>(points_count / static_cast<float>(threads) + 1.0);
    }

};


class CudaTimer {
    cudaEvent_t start_time, end_time;

public:
    CudaTimer() {
        cudaEventCreate(&start_time);
        cudaEventCreate(&end_time);
        cudaEventRecord(start_time);
    }

    float get_elapsed_milliseconds() {
        float milliseconds_time;
        cudaDeviceSynchronize();
        cudaEventRecord(end_time);
        cudaEventSynchronize(end_time);
        cudaEventElapsedTime(&milliseconds_time, start_time, end_time);
        return milliseconds_time;
    }
};


template<typename T>
__device__ void copy_pair_kernel(std::pair<T, T>& source_pair, std::pair<T, T>& dest_pair) {
    dest_pair.first = source_pair.first;
    dest_pair.second = source_pair.second;
}


template<typename T>
__device__ bool pair_equal(std::pair<T, T>& pair1, std::pair<T, T>& pair2) {
    return pair1.first == pair2.first && pair1.second == pair2.second;
}


__device__ bool is_point_on_right_side_of_line_kernel(
    const Point& line_start, const Point& line_end, const Point& point
) {
    float point1_x = point.first - line_start.first;
    float point1_y = point.second - line_start.second;
    float point2_x = line_end.first - line_start.first;
    float point2_y = line_end.second - line_start.second;
    return point1_x * point2_y - point1_y * point2_x < 0;
}


__device__ float get_squared_distance_of_point_from_line_kernel(
    const Point& line_start, const Point& line_end, const Point& point
) {
    float point1_x = point.first - line_start.first;
    float point1_y = point.second - line_start.second;
    float point2_x = line_end.first - line_start.first;
    float point2_y = line_end.second - line_start.second;
    float root_of_enumerator = point1_x * point2_y - point1_y * point2_x;
    float enumerator = root_of_enumerator * root_of_enumerator;
    float denominator = point2_x * point2_x + point2_y * point2_y;
    return enumerator / denominator;
}


__global__ void set_initial_segments_kernel(Point* points, Point* segments) {
    extern __shared__ Point shared_elements[];
    Point* min_elements = shared_elements;
    Point* max_elements = shared_elements + blockDim.x;
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid >= initial_points_count) {
        return;
    }
    int local_tid = threadIdx.x;
    copy_pair_kernel(points[global_tid], min_elements[local_tid]);
    copy_pair_kernel(points[global_tid], max_elements[local_tid]);
    for (int d = blockDim.x >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (local_tid >= d || global_tid + d >= initial_points_count) {
            continue;
        }
        if (min_elements[local_tid + d].first < min_elements[local_tid].first) {
            copy_pair_kernel(min_elements[local_tid + d], min_elements[local_tid]);
        }
        if (max_elements[local_tid + d].first > max_elements[local_tid].first) {
            copy_pair_kernel(max_elements[local_tid + d], max_elements[local_tid]);
        }
    }
    if (local_tid != 0) {
        return;
    }
    bool loop_finished = false;
    while (!loop_finished) {
        if (atomicExch(&lock, 1) == 0) {
            if (min_elements[0].first < segments[0].first) {
                copy_pair_kernel(min_elements[0], segments[0]);
            }
            if (max_elements[0].first > segments[1].first) {
                copy_pair_kernel(max_elements[0], segments[1]);
            }
            __threadfence();
            loop_finished = true;
            atomicExch(&lock, 0);
        }
    }
}


__global__ void swap_first_two_segments_kernel(Point* segments) {
    if (threadIdx.x != 0) {
        return;
    }
    float tmp_first, tmp_second;
    tmp_first = segments[1].first;
    tmp_second = segments[1].second;
    copy_pair_kernel(segments[0], segments[1]);
    segments[0].first = tmp_first;
    segments[0].second = tmp_second;
}


__global__ void filter_out_left_side_points_kernel(
    Point* input_points, Point* output_points, Point* line_start, Point* line_end
) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid >= initial_points_count) {
        return;
    }
    if (pair_equal(input_points[global_tid], *line_start) || pair_equal(input_points[global_tid], *line_end)) {
        return;
    }
    if (is_point_on_right_side_of_line_kernel(*line_start, *line_end, input_points[global_tid])) {
        copy_pair_kernel(input_points[global_tid], output_points[atomicAdd(&points_count, 1)]);
    }
}


__global__ void reset_segments_kernel(
    float* segment_max_dists, int* new_segment_indexes, int* segments_count_prefix_sums, Point* segments
) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid < segments_count) {
        // printf("%d (%f, %f)\n", global_tid, segments[global_tid].first, segments[global_tid].second); #TODEL
        segment_max_dists[global_tid] = 0.0;
        new_segment_indexes[global_tid] = -1;
        segments_count_prefix_sums[global_tid] = 0;
    }
}


__global__ void calculate_max_dists_kernel(
    Point* points, Point* segments, int* labels, float* segment_max_dists, float* line_distances
) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid >= points_count) {
        return;
    }
    line_distances[global_tid] = get_squared_distance_of_point_from_line_kernel(
        segments[labels[global_tid]], segments[labels[global_tid] + 1], points[global_tid]
    );
    atomicMax(&segment_max_dists[labels[global_tid]], line_distances[global_tid]);
}


__global__ void calculate_new_segment_indexes_kernel(
    Point* points, int* labels, float* segment_max_dists, float* line_distances, int* new_segment_indexes
) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid >= points_count || line_distances[global_tid] <= 0.0) {
        return;
    }
    if (line_distances[global_tid] == segment_max_dists[labels[global_tid]]) {
        atomicMax(&new_segment_indexes[labels[global_tid]], global_tid);
    }
}


__global__ void calculate_points_sides_kernel(
    Point* points, Point* segments, int* labels, float* line_distances, int* new_segment_indexes,
    bool* is_on_left_side, bool* is_on_right_side
) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid >= points_count) {
        return;
    }
    is_on_left_side[global_tid] = false;
    is_on_right_side[global_tid] = false;
    if (line_distances[global_tid] <= 0.0) {
        return;
    }
    int point_of_new_segment_index = new_segment_indexes[labels[global_tid]];
    if (point_of_new_segment_index < 0 || point_of_new_segment_index == global_tid) {
        return;
    }
    if(is_point_on_right_side_of_line_kernel(
        segments[labels[global_tid]], points[point_of_new_segment_index], points[global_tid]
    )) {
        is_on_left_side[global_tid] = true;
        hull_changed = true;
    } else if (is_point_on_right_side_of_line_kernel(
        points[point_of_new_segment_index], segments[labels[global_tid] + 1], points[global_tid]
    )) {
        is_on_right_side[global_tid] = true;
        hull_changed = true;
    }
}


__global__ void calculate_prefix_sums_kernel(int* indexes, int* prefix_sums) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid >= segments_count) {
        return;
    }
    if (global_tid == 0) {
        prefix_sums[0] = (indexes[0] >= 0);
        for (int i = 1; i < segments_count; i++) {
            prefix_sums[i] = prefix_sums[i - 1] + (indexes[i] >= 0);
        }
    }
}


__global__ void move_labels_kernel(
    int* labels, int* segments_count_prefix_sums, bool* is_on_left_side, bool* is_on_right_side
) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid >= points_count) {
        return;
    }
    int prefix_sum = labels[global_tid] != 0 ? segments_count_prefix_sums[labels[global_tid] - 1] : 0;
    if (is_on_left_side[global_tid]) {
        labels[global_tid] = labels[global_tid] + prefix_sum;
    } else if (is_on_right_side[global_tid]) {
        labels[global_tid] = labels[global_tid] + prefix_sum + 1;
    }
}


__global__ void calculate_new_segments_kernel(
    Point* points, Point* segments, Point* next_segments, int* new_segment_indexes, int* segments_count_prefix_sums
) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid >= segments_count) {
        return;
    }
    int prefix_sum = global_tid != 0 ? segments_count_prefix_sums[global_tid - 1] : 0;
    copy_pair_kernel(segments[global_tid], next_segments[global_tid + prefix_sum]);
    int point_of_new_segment_index = new_segment_indexes[global_tid];
    if (point_of_new_segment_index >= 0) {
        copy_pair_kernel(points[point_of_new_segment_index], next_segments[global_tid + prefix_sum + 1]);
    }
    if (global_tid == segments_count - 1) {
        next_segments_count = segments_count + segments_count_prefix_sums[segments_count - 1];
    }
}


__global__ void update_points_and_labels_kernel(
    Point* points, int* labels, Point* next_points, int* next_labels, bool* is_on_left_side, bool* is_on_right_side
) {
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_tid >= points_count) {
        return;
    }
    if (is_on_left_side[global_tid] == false && is_on_right_side[global_tid] == false) {
        return;
    }
    int index = atomicAdd(&next_points_count, 1);
    copy_pair_kernel(points[global_tid], next_points[index]);
    next_labels[index] = labels[global_tid];
}


Point* get_initialized_segments(int points_count) {
    Point* h_segments = (Point*)malloc(sizeof(Point) * 2);
    h_segments[0] = Point(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    h_segments[1] = Point(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    return h_segments;
}


void initialize_managed_memory(int host_points_count) {
    initial_points_count = host_points_count;
    points_count = 0;
    next_points_count = 0;
    segments_count = 2;
    next_segments_count = 0;
    hull_changed = false;
}


std::vector<Point> calculate_right_side_convex_hull_points(KernelOptions options, DeviceData data) {
    filter_out_left_side_points_kernel<<<options.blocks, options.threads>>>(
        data.d_initial_points, data.d_side_points, &data.d_initial_segments[0], &data.d_initial_segments[1]
    );
    cudaMemcpy(
        data.d_side_segments, data.d_initial_segments, sizeof(Point) * segments_count, cudaMemcpyDeviceToDevice
    );
    do {
        hull_changed = false;
        cudaDeviceSynchronize();
        reset_segments_kernel<<<1, segments_count>>>(
            data.d_segment_max_dists, data.d_new_segment_indexes, data.d_segments_count_prefix_sums, data.d_side_segments
        );
        calculate_max_dists_kernel<<<options.blocks, options.threads>>>(
            data.d_side_points, data.d_side_segments, data.d_labels, data.d_segment_max_dists, data.d_line_distances
        );
        calculate_new_segment_indexes_kernel<<<options.blocks, options.threads>>>(
            data.d_side_points, data.d_labels, data.d_segment_max_dists, data.d_line_distances,
            data.d_new_segment_indexes
        );
        calculate_points_sides_kernel<<<options.blocks, options.threads>>>(
            data.d_side_points, data.d_side_segments, data.d_labels, data.d_line_distances,
            data.d_new_segment_indexes, data.d_is_on_left_side, data.d_is_on_right_side
        );
        calculate_prefix_sums_kernel<<<1, segments_count>>>(
            data.d_new_segment_indexes, data.d_segments_count_prefix_sums
        );
        calculate_new_segments_kernel<<<1, segments_count>>>(
            data.d_side_points, data.d_side_segments, data.d_next_side_segments, data.d_new_segment_indexes,
            data.d_segments_count_prefix_sums
        );
        move_labels_kernel<<<options.blocks, options.threads>>>(
            data.d_labels, data.d_segments_count_prefix_sums, data.d_is_on_left_side, data.d_is_on_right_side
        );
        update_points_and_labels_kernel<<<options.blocks, options.threads>>>(
            data.d_side_points, data.d_labels, data.d_next_side_points, data.d_next_labels, data.d_is_on_left_side,
            data.d_is_on_right_side
        );
        cudaDeviceSynchronize();
        points_count = next_points_count;
        next_points_count = 0;
        segments_count = next_segments_count;
        next_segments_count = 0;
        options.update_points_count(points_count);
        data.swap_side_points_pointers();
        data.swap_side_segments_pointers();
        data.swap_labels_pointers();
    } while (hull_changed);
    Point* h_segments = (Point*)malloc(sizeof(Point) * segments_count);
    cudaMemcpy(h_segments, data.d_side_segments, sizeof(Point) * segments_count, cudaMemcpyDeviceToHost);
    return std::vector<Point>(h_segments, h_segments + segments_count);
}

std::vector<Point> calculate_convex_hull_points(KernelOptions options, DeviceData data, int points_count) {
    initialize_managed_memory(points_count);
    data.reset_variables();
    cudaDeviceSynchronize();
    set_initial_segments_kernel<<<options.blocks, options.threads, options.shared_mem>>>(
        data.d_initial_points, data.d_initial_segments
    );
    std::vector<Point> convex_hull_points = calculate_right_side_convex_hull_points(options, data);
    initialize_managed_memory(points_count);
    data.reset_variables();
    cudaDeviceSynchronize();
    swap_first_two_segments_kernel<<<1, 1>>>(data.d_initial_segments);
    std::vector<Point> left_side_points = calculate_right_side_convex_hull_points(options, data);
    convex_hull_points.insert(
    convex_hull_points.end(), left_side_points.begin() + 1, left_side_points.end() - 1
    );
    convex_hull_points.push_back(convex_hull_points[0]);
    return convex_hull_points;
}


ConvexHullTime compute_quickhull_gpu_with_time(std::vector<Point>& h_points, bool include_data_transfer_time) {
    KernelOptions kernel_options(h_points.size(), /*threads_count=*/1024);
    Point* h_segments = get_initialized_segments(h_points.size());
    DeviceData device_data(h_points, h_segments, /*segments_count=*/2);
    if (!include_data_transfer_time) {
        device_data.init_memory();
    }
    CudaTimer timer;
    if (include_data_transfer_time) {
        device_data.init_memory();
    }
    std::vector<Point> points_of_convex_hull = calculate_convex_hull_points(
        kernel_options, device_data, h_points.size()
    );
    float milliseconds_time = timer.get_elapsed_milliseconds();
    device_data.free_memory();
    free(h_segments);
    ConvexHull convex_hull(h_points, points_of_convex_hull);
    return ConvexHullTime(convex_hull, milliseconds_time);
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


Point get_point_with_max_y_coordinate(const std::vector<Point>& points) {
    Point best_point = points[0];
    for (int i = 1; i < points.size(); i++) {
        if (points[i].second > best_point.second) {
            best_point = points[i];
        }
        if (points[i].second == best_point.second && points[i].first < best_point.first) {
            best_point = points[i];
        }
    }
    return best_point;
}


void compute_recursive_quickhull_cpu(
    const Point& line_start, const Point& line_end, const std::vector<Point>& points, std::vector<Point>& chosen_points
) {
    if (points.size() == 0) {
        return;
    }
    Point farthest_point = get_farthest_point_from_line(line_start, line_end, points);
    std::vector<Point> side1_points, side2_points;
    side1_points.reserve(points.size());
    side2_points.reserve(points.size());
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
    Point initial_point = get_point_with_max_y_coordinate(points);
    std::sort(points.begin(), points.end(), [initial_point](const Point& point1, const Point& point2) {
        if (point1 == initial_point) {
            return true;
        }
        return point2 != initial_point && is_point_on_right_side_of_line(initial_point, point2, point1);
    });
    std::vector<Point> chosen_points;
    chosen_points.push_back(points[0]);
    for (int i = 1; i < points.size(); i++) {
        while (chosen_points.size() > 1 && is_point_on_right_side_of_line(
                chosen_points[chosen_points.size() - 2],
                chosen_points[chosen_points.size() - 1],
                points[i])) {
            chosen_points.pop_back();
        }
        chosen_points.push_back(points[i]);
    }
    chosen_points.push_back(points[0]);
    return ConvexHull(points, chosen_points);
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