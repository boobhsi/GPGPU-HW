CC=g++
NVCC=nvcc
CXXFLAGS= -std=c++11
CUDAFLAGS= -std=c++11
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-I/usr/local/cuda/include

all: VideoGen clean

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) -x cu -I. -dc $< -o $@

# lab1.o: lab1.cu
#	$(NVCC) $(CUDAFLAGS) lab1.cu -dc -dlink

VideoGen: lab1.o main.o
	$(NVCC) $(CUDAFLAGS) lab1.o main.o -o VideoGen

# link.o: lab1.o main.o
#	$(NVCC) $(CUDAFLAGS) -dlink lab1.o main.o -o link.o -lcudart

#VideoGen: link.o
#	$(NVCC) $(CUDAFLAGS) link.o -o VideoGen


clean:
	rm -rf *.o

.PHONY: all
