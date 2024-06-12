#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
ifeq ($(VENDOR), acpp)
CC        = acpp
else ifeq ($(VENDOR), intel-llvm)
CC        = clang++
else 
CC	  = icpx 
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

source = src/main.cpp src/parallel-bench.cpp src/vectorization-bench.cpp src/timer.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) $(KERNEL_DIM) -std=c++17 -Wall -DREDUCTION_IN_SYCL -DTYPE=float

ifeq ($(VENDOR), acpp)
CFLAGS += -DHIPSYCL --hipsycl-platform=cpu  -fopenmp
else
CFLAGS += -fsycl -DDPCPP
endif

# Linker Flags
LDFLAGS = 


# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g 
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -Ofast
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp src/parallel-bench.hpp src/vectorization-bench.hpp src/timer.hpp 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(obj)

