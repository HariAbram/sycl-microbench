#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <omp.h>


void parallel_for_omp(int size, int block_size);

void parallel_for_omp_nested(int size, int block_size);

void atomic_reduction_omp( int size, int block_size);

void reduction_without_atomics_omp(int size, int tile_size);

void barrier_test_omp(int size, int block_size);

void kernel_computation();
///////////////
