CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcusparse -fopenmp
CXX = g++
NVCC = nvcc
HEADERS = myHeaders.cuh
NVCCFLAGS = -O0 -std=c++17 -Wno-deprecated-gpu-targets $(INCLUDES)
LDFLAGS = -Llibtorch/lib #-Wl,-rpath,libtorch/lib
INCLUDES = -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include
LIBS = -ltorch -ltorch_cpu -lc10
LIBTORCH = $(INCLUDES) $(LDFLAGS) $(LIBS)

all: spmm_blockedell_example functions.o test.o

spmm_blockedell_example: spmm_blockedell_example.cu functions.o test.o $(HEADERS)
	$(NVCC) $(NVCCFLAGS) spmm_blockedell_example.cu functions.o test.o \
    -Xcompiler -fopenmp \
    -I/opt/cuda/include \
    -Ilibtorch/include \
    -Ilibtorch/include/torch/csrc/api/include \
    -Llibtorch/lib \
    -Xlinker -rpath -Xlinker libtorch/lib \
    -ltorch -ltorch_cpu -lc10 \
    -lcusparse \
    -o spmm_blockedell_example

functions.o: functions.cu $(HEADERS)
	$(NVCC) $(CXXFLAGS) -c functions.cu $(LIBTORCH) -o $@

test.o: test.cu $(HEADERS)
	$(NVCC) $(CXXFLAGS) -c test.cu $(LIBTORCH) -o $@
clean:
	rm -f spmm_blockedell_example *.o

test:
	@echo "\n==== SpMM BLOCKED ELL Test ====\n"
	./spmm_blockedell_example

.PHONY: clean all test
