#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "convex_hull_display.h"

ConvexHull* ConvexHullDisplay::convex_hull = NULL;
const int SCREEN_WIDTH = 600, SCREEN_HEIGHT = 600;

int x_coord_to_screen_coord(float coord_x, float min_value, float max_value) {
    return (coord_x - min_value) / (max_value - min_value) * SCREEN_WIDTH;
}


int y_coord_to_screen_coord(float coord_y, float min_value, float max_value) {
    return (coord_y - min_value) / (max_value - min_value) * SCREEN_HEIGHT;
}


void generate_point_on_screen(Point point, float min_value = -1.0, float max_value = 1.0) {
    int coord_x = x_coord_to_screen_coord(point.first, min_value, max_value);
    int coord_y = y_coord_to_screen_coord(point.second, min_value, max_value);
    glVertex2i(coord_x, glutGet(GLUT_WINDOW_HEIGHT) - coord_y);
}


void generate_line_on_screen(Point point1, Point point2, float min_value = -1.0, float max_value = 1.0) {
    int coord1_x = x_coord_to_screen_coord(point1.first, min_value, max_value);
    int coord1_y = y_coord_to_screen_coord(point1.second, min_value, max_value);
    int coord2_x = x_coord_to_screen_coord(point2.first, min_value, max_value);
    int coord2_y = y_coord_to_screen_coord(point2.second, min_value, max_value);
    glVertex2i(coord1_x, glutGet(GLUT_WINDOW_HEIGHT) - coord1_y);
    glVertex2i(coord2_x, glutGet(GLUT_WINDOW_HEIGHT) - coord2_y);
}


void ConvexHullDisplay::draw_convex_hull() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);
    for (int i = 0; i < convex_hull->first.size(); i++) {
        generate_point_on_screen(convex_hull->first[i]);
    }
    glEnd();
    glBegin(GL_LINES);
    glColor3f(0.0, 0.0, 0.0);
    for (int i = 1; i < convex_hull->second.size(); i++) {
        generate_line_on_screen(convex_hull->second[i - 1], convex_hull->second[i]);
    }
    glEnd();
    glFlush();
}


void ConvexHullDisplay::show_on_screen() {
    if (convex_hull == NULL) {
        return;
    }
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
    glutInitWindowPosition(200, 200);
    glutCreateWindow("A drawing of computed convex hull");
    glClearColor(1.0, 1.0, 1.0, 0.0);
    glMatrixMode(GL_PROJECTION);
    gluOrtho2D(0.0, SCREEN_WIDTH, 0.0, SCREEN_HEIGHT);
    glPointSize(8.0);
    glutDisplayFunc(&ConvexHullDisplay::draw_convex_hull);
    glutMainLoop();
}

