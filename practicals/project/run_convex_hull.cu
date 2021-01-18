/* A program that aims to compare the performance of different 2D convex hull computation methods. Additionally, it is
 * capable of drawing a computed convex hull on the screen.
 *
 * Positional arguments of the program:
 *  - hull algorithm: defines which algorithm should be used to compute a convex hull. It should be equal to one of the
 *                    following values: 'quickhull_cpu', 'quickhull_gpu', 'graham_cpu'
 *  - points counts: The number of points that will be used to calculate a convex hull
 *  - simulations count: The number of times a convex hull computation algorithm will be run
 *  - seed: Random seed used to draw points
 *  - output file: A location of a file where points of the last computed convex hull should be saved. If it is equal
 *                 to 0, a convex hull is not saved.
 *  - display convex hull: Whether to display the last computed convex hull on the screen
 *
 * Example usage:
 *  ./run_convex_hull quickhull_cpu 10 100 1 123
 */
#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include "common.h"
#include "convex_hull_computation.h"
#include "convex_hull_display.h"


float random_float(float min_value, float max_value) {
    float zero_one_range_value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return zero_one_range_value * (max_value - min_value) + min_value;
}


std::vector<Point> random_points(int points_count) {
    std::vector<Point> points;
    while (points.size() < points_count) {
        float x_coord = random_float(-1.0, 1.0);
        float y_coord = random_float(-1.0, 1.0);
        if (x_coord * x_coord + y_coord * y_coord <= 1.0) {
            points.push_back(Point(x_coord, y_coord));
        }
    }
    return points;
}


ConvexHull run_simulations(const std::string& hull_algorithm, const int points_count, const int simulations_count) {
    std::vector<float> run_times;
    ConvexHullTime convex_hull_run_time;
    for (int i = 0; i < simulations_count; i++) {
        std::vector<Point> points = random_points(points_count);
        convex_hull_run_time = compute_convex_hull_with_time(hull_algorithm, points, false);
        run_times.push_back(convex_hull_run_time.second);
    }
    float total_run_time = std::accumulate(run_times.begin(), run_times.end(), 0.0);
    float mean_run_time = total_run_time / run_times.size();
    std::vector<float> squared_run_times;
    squared_run_times.resize(run_times.size());
    std::transform(run_times.begin(), run_times.end(), squared_run_times.begin(), [](float x) { return x * x; });
    float total_squared_run_time = std::accumulate(squared_run_times.begin(), squared_run_times.end(), 0.0);
    float var_run_time = (total_squared_run_time / run_times.size() - mean_run_time * mean_run_time);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Computation time mean: " << mean_run_time << " ms\n";
    std::cout << "Computation time variance: " << var_run_time << " ms\n";
    return convex_hull_run_time.first;
}


bool arguments_valid(int argc, char **argv) {
    if (argc != 7) {
        std::cout << "Error: expected 7 arguments to be passed, got " << argc << "\n";
        return false;
    }
    std::string hull_algorithm = argv[1];
    if (hull_algorithm != "quickhull_cpu" && hull_algorithm != "quickhull_gpu" && hull_algorithm != "graham_cpu") {
        std::cout << "Error: expected hull algorithm to be equal to one of the following values: "
                  << "'quickhull_cpu', 'quickhull_gpu', 'graham_cpu'" << hull_algorithm << "'\n";
        return false;
    }
    int points_count = atoi(argv[2]);
    if (points_count < 3) {
        std::cout << "Error: expected the number of points to be greater or equal to 3, got " << points_count << "\n";
        return false;
    }
    return true;
}

void save_convex_hull(const ConvexHull& convex_hull, const std::string& output_file) {
    std::ofstream file_stream;
    file_stream.open(output_file);
    file_stream << std::fixed << std::setprecision(3) << convex_hull.first.size() << "\n";
    for (const Point& point : convex_hull.first) {
        file_stream << "(" << point.first << ", " << point.second << ") ";
    }
    file_stream << "\n" << convex_hull.second.size() << "\n";
    for (const Point& point : convex_hull.second) {
        file_stream << "(" << point.first << ", " << point.second << ") ";
    }
    file_stream << "\n";
    int hull_index = 0;
    for (int arr_iteration = 0; arr_iteration < 2; arr_iteration++) {
        for (int i = 0; i < convex_hull.first.size(); i++) {
            if (hull_index == convex_hull.second.size()) {
                break;
            }
            if (convex_hull.first[i] == convex_hull.second[hull_index]) {
                file_stream << i << " ";
                hull_index++;
            }
        }
    }
    file_stream << "\n";
    file_stream.close();
}

int main(int argc, char **argv) {
    if (!arguments_valid(argc, argv)) {
        return 1;
    }
    std::string hull_algorithm = argv[1];
    int points_count = atoi(argv[2]);
    int simulations_count = atoi(argv[3]);
    int seed = atoi(argv[4]);
    std::string output_file = argv[5];
    bool hull_should_be_drawn = atoi(argv[6]);
    std::cout << "Running calculations on " << hull_algorithm << " (points count: " << points_count
        << ", simulations count: " << simulations_count << ")\n";
    srand(seed);
    ConvexHull convex_hull = run_simulations(hull_algorithm, points_count, simulations_count);
    if (output_file != "0") {
        std::cout << "Saving computed convex hull to the output file: '" << output_file << "'\n";
        save_convex_hull(convex_hull, output_file);
    }
    if (hull_should_be_drawn) {
        std::cout << "Drawing last computed convex hull on the screen\n";
        glutInit(&argc, argv);
        ConvexHullDisplay::convex_hull = &convex_hull;
        ConvexHullDisplay::show_on_screen();
    }
    return 0;
}
