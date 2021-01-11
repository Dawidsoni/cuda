/* A program that aims to compare the performance of different 2D convex hull computation methods. Additionally, it is
 * capable of drawing a computed convex hull on the screen.
 *
 * Positional arguments of the program:
 *  - hull algorithm: defines which algorithm should be used to compute a convex hull. It should be equal to one of the
 *                    following values: 'quickhull_cpu', 'quickhull_gpu', 'graham_cpu'
 *  - points counts: The number of points that will be used to calculate a convex hull
 *  - simulations count: The number of times a convex hull computation algorithm will be run
 *  - draw convex hull: Whether to show a computed convex hull on a screen (if the simulations count is greater than 1,
 *                      the last computed convex hull is shown on the screen)
 *
 * Example usage:
 *  ./run_convex_hull quickhull_cpu 10 100 1
 */
#include <iostream>
#include <vector>
#include <numeric>
#include <stdlib.h>
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


ConvexHull run_simulations(std::string& hull_algorithm, int points_count, int simulations_count) {
    std::vector<float> run_times;
    for (int i = 0; i < simulations_count; i++) {
        run_times.push_back(1.0);
    }
    float mean = std::accumulate(run_times.begin(), run_times.end(), 0.0) / run_times.size();
    std::vector<Point> xxx {Point(-0.9, -0.9), Point(0.9, -0.9), Point(-0.9, 0.9), Point(0.9, 0.9)};
    std::vector<Point> yyy { xxx[0], xxx[1], xxx[2], xxx[0]};
    return ConvexHull(xxx, yyy);
}


int main(int argc, char **argv) {
    if (argc != 5) {
        std::cout << "Error: expected 5 arguments to be passed, got " << argc << "\n";
        return 1;
    }
    std::string hull_algorithm = argv[1];
    if (hull_algorithm != "quickhull_cpu" && hull_algorithm != "quickhull_gpu" && hull_algorithm != "graham_cpu") {
        std::cout << "Error: expected hull algorithm to be equal to one of the following values: "
            << "'quickhull_cpu', 'quickhull_gpu', 'graham_cpu'" << hull_algorithm << "'\n";
        return 1;
    }
    int points_count = atoi(argv[2]);
    int simulations_count = atoi(argv[3]);
    bool hull_should_be_drawn = atoi(argv[4]);
    std::cout << "Running calculations on " << hull_algorithm << " (points count: " << points_count
        << ", simulations count: " << simulations_count << ")\n";
    srand(time(NULL));
    ConvexHull convex_hull = run_simulations(hull_algorithm, points_count, simulations_count);
    if (hull_should_be_drawn) {
        std::cout << "Drawing last computed convex hull on the screen\n";
        glutInit(&argc, argv);
        ConvexHullDisplay::convex_hull = &convex_hull;
        ConvexHullDisplay::show_on_screen();
    }
    return 0;
}
