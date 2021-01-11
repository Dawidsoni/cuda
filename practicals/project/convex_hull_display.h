#ifndef CUDA_PRAC1_CONVEX_HULL_DISPLAY_H
#define CUDA_PRAC1_CONVEX_HULL_DISPLAY_H

#include "common.h"
#include <GL/glut.h>

class ConvexHullDisplay {
private:
    static void draw_convex_hull();
public:
    static ConvexHull* convex_hull;
    static void show_on_screen();
};


#endif //CUDA_PRAC1_CONVEX_HULL_DISPLAY_H
