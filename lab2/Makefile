CC=g++
NVCC=nvcc
CXXFLAGS= -std=c++11
CUDAFLAGS= -std=c++11 -arch sm_30 -O2
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-I/usr/local/cuda/include

all: VideoGen clean

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) $(LIBDIRS) $(INCDIRS) -x cu -dc $< -o $@

# lab1.o: counting.cu
#	$(NVCC) $(CUDAFLAGS) lab1.cu -dc -dlink

VideoGen: counting.o main.o
	$(NVCC) $(CUDAFLAGS) $(INCDIRS) $(INDCIRS) counting.o main.o -o HW2

# link.o: lab1.o main.o
#	$(NVCC) $(CUDAFLAGS) -dlink lab1.o main.o -o link.o -lcudart

#VideoGen: link.o
#	$(NVCC) $(CUDAFLAGS) link.o -o VideoGen


clean:
	rm -rf *.o

.PHONY: all
