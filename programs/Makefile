all: test.o docTests.o classes.o COO.o

docTests.o:
	g++ -std=c++17 docTests.cc -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include -Llibtorch/lib -Wl,-rpath,libtorch/lib -ltorch -ltorch_cpu -lc10 -o docTests.o
test.o:
	g++ -std=c++17 test.cc -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include -Llibtorch/lib -Wl,-rpath,libtorch/lib -ltorch -ltorch_cpu -lc10 -o test.o
classes.o:
	g++ -g -Wall classes.cc -o classes.o
COO.o:
	g++ -g -Wall -fopenmp COO.cc -o COO.o

clean:
	rm -rf *.o
