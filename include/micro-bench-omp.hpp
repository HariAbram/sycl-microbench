#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <omp.h>


void parallel_for_omp(int size, bool print, int iter);

void parallel_for_omp_nested(int size, bool print, int iter);

void atomics_omp( int size, bool print, int iter);

void reduction_omp(int size, bool print, int iter);

void barrier_test_omp(int size, bool print, int iter);

void kernel_computation();
///////////////
