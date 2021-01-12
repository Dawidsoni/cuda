
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcurand

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

all:	compare_computation_speed

compare_computation_speed:	 compare_computation_speed.cu Makefile
	nvcc compare_computation_speed.cu -o compare_computation_speed $(INC) $(NVCCFLAGS) $(LIB)

clean:
	rm -f compare_computation_speed
