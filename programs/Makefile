CXX = g++
CXXFLAGS = -g -fopenmp -std=c++17
LDFLAGS = -Llibtorch/lib -Wl,-rpath,libtorch/lib
INCLUDES = -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include
LIBS = -ltorch -ltorch_cpu -lc10

LIBTORCH = $(INCLUDES) $(LDFLAGS) $(LIBS)
HEADERS = myHeaders.hpp

TARGETS = test_sparse_conversion.out test_count_zero_blocks.out docTests.out test.out classes.out COO.out find_best_block_size.out

all: $(TARGETS)

find_best_block_size.out: find_best_block_size.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) find_best_block_size.cpp $(LIBTORCH) -o $@

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
