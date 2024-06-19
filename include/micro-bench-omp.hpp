#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>

void std_memory_alloc(int size, int iter, bool print);

void parallel_for_omp(int size, bool print, int iter);

void parallel_for_omp_nested(int size, bool print, int iter);

void atomics_omp( int size, bool print, int iter);

void reduction_omp(int size, bool print, int iter);

void barrier_test_omp(int size, bool print, int iter);
///////////////
