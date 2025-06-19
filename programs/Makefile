CXX = g++
NVCC = nvcc
# CXXFLAGS = -g -fopenmp -std=c++17
# ATTENTION: watcha out for the -march=native flag!
INCLUDES = -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include
LIBS = -ltorch -ltorch_cpu -lc10

CXXFLAGS = -O3 -march=native -mavx2 -mfma -fopenmp -std=c++17
NVCCFLAGS = -O0 -std=c++17 -Wno-deprecated-gpu-targets $(INCLUDES)

LDFLAGS = -Llibtorch/lib -Wl,-rpath,libtorch/lib
CUDA_LDFLAGS = -Xlinker -rpath -Xlinker libtorch/lib -Llibtorch/lib

LIBTORCH = $(INCLUDES) $(LDFLAGS) $(LIBS)
CUDA_LIBTORCH = $(INCLUDES) $(CUDA_LDFLAGS) $(LIBS)
HEADERS = myHeaders.hpp
CUDA_HEADERS = cudaHeaders.hpp

TARGETS = test_sparse_conversion.o test_count_zero_blocks.o docTests.o test.o classes.o COO.o findBellParams.o functions.o findBellParamsCUDA cudaFunctions.o cudaFindBellParams.o constants.o

all: $(TARGETS)

findBellParamsCUDA: cudaFindBellParams.o cudaFunctions.o
	$(NVCC) $(NVCCFLAGS) cudaFindBellParams.o cudaFunctions.o $(CUDA_LIBTORCH) -o $@

constants.o: constants.cu constants.hpp
	$(NVCC) $(NVCCFLAGS) -c constants.cu -o $@

cudaFunctions.o: cudaFunctions.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCCFLAGS) -c cudaFunctions.cu -o $@

cudaFindBellParams.o: cudaFindBellParams.cu $(CUDA_HEADERS)
	$(NVCC) $(NVCCFLAGS) -c cudaFindBellParams.cu -o $@

functions.o: functions.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c functions.cpp $(LIBTORCH) -o $@

findBellParams.o: findBellParams.cpp functions.o $(HEADERS)
	$(CXX) $(CXXFLAGS) findBellParams.cpp functions.o $(LIBTORCH) -o $@

test_sparse_conversion.o: test_sparse_conversion.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) test_sparse_conversion.cpp $(LIBTORCH) -o $@

test_count_zero_blocks.o: test_count_zero_blocks.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) test_count_zero_blocks.cpp $(LIBTORCH) -o $@

docTests.o: docTests.cc $(HEADERS)
	$(CXX) -std=c++17 docTests.cc $(LIBTORCH) -o $@

test.o: test.cc $(HEADERS)
	$(CXX) -std=c++17 test.cc $(LIBTORCH) -o $@

classes.o: classes.cc $(HEADERS)
	$(CXX) -g -Wall classes.cc -o $@

COO.o: COO.cc $(HEADERS)
	$(CXX) -g -Wall -fopenmp COO.cc -o $@

clean:
	rm -f $(TARGETS)
