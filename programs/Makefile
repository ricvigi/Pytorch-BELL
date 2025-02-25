all: test.o docTests.o

docTests.o:
	g++ -std=c++17 docTests.cc -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include -Llibtorch/lib -Wl,-rpath,libtorch/lib -ltorch -ltorch_cpu -lc10 -o docTests.o
test.o:
	g++ -std=c++17 test.cc -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include -Llibtorch/lib -Wl,-rpath,libtorch/lib -ltorch -ltorch_cpu -lc10 -o test.o
classes.o:
	g++ classes.cc -o classes.o

clean:
	rm -rf *.o
