#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>

using namespace cl;

double dealy_time();

void host_memory_alloc(sycl::queue &Q, int size);

void shared_memory_alloc(sycl::queue &Q, int size);

void device_memory_alloc(sycl::queue &Q, int size);

void kernel_offloading_duration(sycl::queue &Q, int size);

void range_1d_with_usm(sycl::queue &Q, int size, int block_size);

void range_2d_with_usm(sycl::queue &Q, int size, int block_size);

void ndrange_1d_with_usm(sycl::queue &Q, int size, int block_size);

void ndrange_2d_with_usm(sycl::queue &Q, int size, int block_size);

void atomic_test_with_usm_and_range_1dim(sycl::queue &Q, int size, int block_size);

void reduction_without_atomics(sycl::queue &Q, int size, int tile_size);

void barrier_test(sycl::queue &Q, int size, int block_size);

void local_barrier_test(sycl::queue &Q, int size, int block_size);

///////////////

void range_1d_with_buff_acc(sycl::queue &Q, int size, int block_size);

void range_2d_with_buff_acc(sycl::queue &Q, int size, int block_size);

void ndrange_1d_with_buff_acc(sycl::queue &Q, int size, int block_size);

void ndrange_2d_with_buff_acc(sycl::queue &Q, int size, int block_size);