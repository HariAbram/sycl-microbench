#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>

using namespace cl;

void vec_add_range_usm(sycl::queue &Q, int size);

void vec_add_range_buff_acc(sycl::queue &Q, int size);

void vec_add_ndrange_usm(sycl::queue &Q, int size, int block_size);

void vec_add_ndrange_buff_acc(sycl::queue &Q, int size, int block_size);

void mat_mul_range_usm(sycl::queue &Q, int size, bool OMP);

void mat_mul_range_buff_acc(sycl::queue &Q, int size, bool OMP);

void mat_mul_ndrange_usm(sycl::queue &Q, int size, int block_size, bool OMP);

void mat_mul_ndrange_buff_acc(sycl::queue &Q, int size, int block_size, bool OMP);

