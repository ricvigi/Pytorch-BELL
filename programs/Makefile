CXX = g++
NVCC = nvcc
# CXXFLAGS = -g -fopenmp -std=c++17
# ATTENTION: watcha out for the -march=native flag!
CXXFLAGS = -O3 -march=native -mavx2 -mfma -fopenmp -std=c++17
NVCCFLAGS = -O3 -std=c++17 -Wno-deprecated-gpu-targets $(INCLUDES)

LDFLAGS = -Llibtorch/lib -Wl,-rpath,libtorch/lib
CUDA_LDFLAGS = -Xlinker -rpath -Xlinker libtorch/lib -Llibtorch/lib

INCLUDES = -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include
LIBS = -ltorch -ltorch_cpu -lc10

LIBTORCH = $(INCLUDES) $(LDFLAGS) $(LIBS)
CUDA_LIBTORCH = $(INCLUDES) $(CUDA_LDFLAGS) $(LIBS)
HEADERS = myHeaders.hpp
CUDA_HEADERS = cudaHeaders.hpp

TARGETS = test_sparse_conversion.out test_count_zero_blocks.out docTests.out test.out classes.out COO.out findBellParams.out functions.out findBellParamsCUDA cudaFunctions.o cudaFindBellParams.o

all: $(TARGETS)

findBellParamsCUDA: cudaFindBellParams.o cudaFunctions.o
	$(NVCC) $(NVCCFLAGS) cudaFindBellParams.o cudaFunctions.o $(CUDA_LIBTORCH) -o $@

cudaFunctions.o: cudaFunctions.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCCFLAGS) -c cudaFunctions.cu -o $@

cudaFindBellParams.o: cudaFindBellParams.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCCFLAGS) -c cudaFindBellParams.cu -o $@

functions.out: functions.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c functions.cpp $(LIBTORCH) -o $@

findBellParams.out: findBellParams.cpp functions.out $(HEADERS)
	$(CXX) $(CXXFLAGS) findBellParams.cpp functions.out $(LIBTORCH) -o $@

test_sparse_conversion.out: test_sparse_conversion.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) test_sparse_conversion.cpp $(LIBTORCH) -o $@

test_count_zero_blocks.out: test_count_zero_blocks.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) test_count_zero_blocks.cpp $(LIBTORCH) -o $@

docTests.out: docTests.cc $(HEADERS)
	$(CXX) -std=c++17 docTests.cc $(LIBTORCH) -o $@

test.out: test.cc $(HEADERS)
	$(CXX) -std=c++17 test.cc $(LIBTORCH) -o $@

classes.out: classes.cc $(HEADERS)
	$(CXX) -g -Wall classes.cc -o $@

COO.out: COO.cc $(HEADERS)
	$(CXX) -g -Wall -fopenmp COO.cc -o $@

clean:
	rm -f $(TARGETS)
