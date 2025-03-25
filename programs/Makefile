all: test_sparse_conversion.o

docTests.o: docTests.cc
	g++ -std=c++17 docTests.cc -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include -Llibtorch/lib -Wl,-rpath,libtorch/lib -ltorch -ltorch_cpu -lc10 -o docTests.o
test.o: test.cc
	g++ -std=c++17 test.cc -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include -Llibtorch/lib -Wl,-rpath,libtorch/lib -ltorch -ltorch_cpu -lc10 -o test.o
classes.o: classes.cc
	g++ -g -Wall classes.cc -o classes.o
COO.o: COO.cc
	g++ -g -Wall -fopenmp COO.cc -o COO.o
test_sparse_conversion.o: test_sparse_conversion.cpp
	g++ -g -fopenmp -std=c++17 test_sparse_conversion.cpp -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include -Llibtorch/lib -Wl,-rpath,libtorch/lib -ltorch -ltorch_cpu -lc10 -o test_sparse_conversion.o
clean:
	rm -rf *.o
