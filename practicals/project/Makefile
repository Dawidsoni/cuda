CC := convex_hull_computation.cu convex_hull_display.cpp run_convex_hull.cu
INCLUDES := -I$(CUDA_HOME)/include -I.
LIBS := -L$(CUDA_HOME)/lib64 -lcudart
OPENGL_FLAGS := -lGL -lGLU -lglut -lglfw
NVCC_FLAGS := -lineinfo -arch=sm_61 --ptxas-options=-v --use_fast_math

all: run_convex_hull

run_convex_hull: $(CC)
	nvcc -std=c++11 -o run_convex_hull $(CC) $(OPENGL_FLAGS) $(INCLUES) $(LIBS) $(NVCC_FLAGS)

clean:
	rm -f run_convex_hull
