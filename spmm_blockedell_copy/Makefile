# Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
CUDA_TOOLKIT := $(shell dirname $$(command -v nvcc))/..
INC          := -I$(CUDA_TOOLKIT)/include
LIBS         := -lcusparse -fopenmp
HEADERS = ../programs/myHeaders.hpp
NVCCFLAGS = -O0 -std=c++17 -Wno-deprecated-gpu-targets $(INCLUDES)
LDFLAGS = -Llibtorch/lib -Wl,-rpath,libtorch/lib
INCLUDES = -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include
LIBS = -ltorch -ltorch_cpu -lc10
LIBTORCH = $(INCLUDES) $(LDFLAGS) $(LIBS)

all: spmm_blockedell_example functions.o

spmm_blockedell_example: spmm_blockedell_example.cpp functions.o $(HEADERS)
	nvcc $(NVCCFLAGS) spmm_blockedell_example.cpp functions.o \
    -Xcompiler -fopenmp \
    -I/opt/cuda/include \
    -Ilibtorch/include \
    -Ilibtorch/include/torch/csrc/api/include \
    -Llibtorch/lib \
    -Xlinker -rpath -Xlinker libtorch/lib \
    -ltorch -ltorch_cpu -lc10 \
    -lcusparse \
    -o spmm_blockedell_example

functions.o: ../programs/functions.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c ../programs/functions.cpp $(LIBTORCH) -o $@
clean:
	rm -f spmm_blockedell_example *.o

test:
	@echo "\n==== SpMM BLOCKED ELL Test ====\n"
	./spmm_blockedell_example

.PHONY: clean all test
