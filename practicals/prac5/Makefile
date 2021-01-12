
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcufft

NVCCFLAGS	:= -lineinfo -arch=sm_35 --ptxas-options=-v --use_fast_math

both:		simpleCUBLAS simpleCUFFT linear_equations

simpleCUBLAS:	simpleCUBLAS.cpp Makefile
		g++ simpleCUBLAS.cpp -o simpleCUBLAS $(INC) $(LIB)

simpleCUFFT:	simpleCUFFT.cu Makefile
		nvcc simpleCUFFT.cu -o simpleCUFFT $(INC) $(NVCCFLAGS) $(LIB)

linear_equations:	linear_equations.cpp Makefile
		nvcc linear_equations.cpp -o linear_equations $(INC) $(NVCCFLAGS) $(LIB)


clean:
		rm -f simpleCUBLAS simpleCUFFT linear_equations
