#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
ifeq ($(VENDOR), acpp)
CXX        = acpp
else ifeq ($(VENDOR), intel-llvm)
CXX       = clang++
else 
CXX	  = icpx 
endif

OPTIMIZE  = yes
DEBUG     = no

#===============================================================================
# Program name & source code list
#===============================================================================

ifeq ($(VENDOR), acpp)
program = bin/main-acpp
else ifeq ($(VENDOR), intel-llvm)
program = bin/main-intel-llvm
else 
program = bin/main-dpcpp
endif

source = src/main.cpp\
         src/parallel-bench-usm.cpp\
         src/parallel-bench-acc.cpp\
         src/kernels.cpp\
         src/utils.cpp\
         src/vectorization-bench.cpp\
         src/timer.cpp\
         src/micro-bench-omp.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CXXFLAGS := $(EXTRA_CFLAGS) $(KERNEL_DIM) -std=c++17 -Wall -DTYPE=float

ifeq ($(VENDOR), acpp)
CXXFLAGS += -DHIPSYCL --hipsycl-platform=cpu  -fopenmp --acpp-targets=omp.accelerated -DACPP
else ifeq ($(VENDOR), intel-llvm)
CXXFLAGS += -fsycl -fopenmp
else 
CXXFLAGS += -fsycl -qopenmp
endif

# Linker Flags
LDFLAGS = 


# Debug Flags
ifeq ($(DEBUG),yes)
  CXXFLAGS  += -g 
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CXXFLAGS += -Ofast
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CXX) $(CXXFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp src/parallel-bench.hpp src/vectorization-bench.hpp src/timer.hpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(obj)

