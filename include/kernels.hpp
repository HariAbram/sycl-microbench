#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <algorithm>
#include <string>


#ifndef TYPE
#define TYPE double
#endif

#include "../include/timer.hpp"
#include "../include/parallel-bench.hpp"

using namespace cl;

/// init arrays

void init_arrays(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global);

void init_arrays(sycl::queue &Q, sycl::buffer<TYPE, 1>  m_buff, sycl::buffer<TYPE, 1> a_buff, sycl::range<1> global);

/// copy

void kernel_copy(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global);

void kernel_copy(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global, sycl::range<1> local);

void kernel_copy(sycl::queue &Q, sycl::buffer<TYPE, 1> m_buff, sycl::buffer<TYPE, 1> a_buff, sycl::range<1> global);

void kernel_copy(TYPE* m, TYPE* a, int size);

/// parallel

void kernel_parallel_1(sycl::queue &Q, TYPE* sum, sycl::range<1> global);

void kernel_parallel_1(sycl::queue &Q, TYPE* sum, sycl::range<1> global, sycl::range<1> local);

void kernel_parallel_1(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<1> global);

void kernel_parallel_1(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<1> global, sycl::range<1> local);


void kernel_parallel_2(sycl::queue &Q, TYPE* sum, sycl::range<2> global);

void kernel_parallel_2(sycl::queue &Q, TYPE* sum, sycl::range<2> global, sycl::range<2> local);

void kernel_parallel_2(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<2> global);

void kernel_parallel_2(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<2> global, sycl::range<2> local);



#endif