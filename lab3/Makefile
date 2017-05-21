CC=g++
NVCC=nvcc
CXXFLAGS= -std=c++11
CUDAFLAGS= -std=c++11 -arch sm_30 -O2
LIBDIRS=-L/usr/local/cuda/lib64
INCDIRS=-I/usr/local/cuda/include

all: sc.out clean

%.o: %.cu
	$(NVCC) $(CUDAFLAGS) $(LIBDIRS) $(INCDIRS) -x cu -dc $< -o $@

# lab1.o: counting.cu
#	$(NVCC) $(CUDAFLAGS) lab1.cu -dc -dlink

sc.out: pgm.o main.o lab3.o
	$(NVCC) $(CUDAFLAGS) $(INCDIRS) $(INDCIRS) pgm.o main.o lab3.o -o sc.out

# link.o: lab1.o main.o
#	$(NVCC) $(CUDAFLAGS) -dlink lab1.o main.o -o link.o -lcudart

#VideoGen: link.o
#	$(NVCC) $(CUDAFLAGS) link.o -o VideoGen


clean:
	rm -rf *.o

.PHONY: all
