#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>

using namespace cl;

double delay_time();

// memory allocation 

void host_memory_alloc(sycl::queue &Q, int size);

void shared_memory_alloc(sycl::queue &Q, int size);

void device_memory_alloc(sycl::queue &Q, int size);

//parallelization

void range_with_usm(sycl::queue &Q, int size, int dim);

void range_with_buff_acc(sycl::queue &Q, int size, int dim);

void nd_range_with_usm(sycl::queue &Q, int size, int block_size ,int dim);

void nd_range_with_buff_acc(sycl::queue &Q, int size, int block_size ,int dim);

//reduction

void reduction_with_atomics_buf_acc(sycl::queue &Q, int size);

void reduction_with_atomics_usm(sycl::queue &Q, int size);

void reduction_with_buf_acc(sycl::queue &Q, int size, int block_size);

// barriers

void global_barrier_test_usm(sycl::queue &Q, int size, int block_size);

void global_barrier_test_buff_acc(sycl::queue &Q, int size, int block_size);

void local_barrier_test_usm(sycl::queue &Q, int size, int block_size);

void local_barrier_test_buff_acc(sycl::queue &Q, int size, int block_size);



