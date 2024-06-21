#ifndef UTILS_H
#define UTILS_H

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
#include <iomanip>

#ifndef TYPE
#define TYPE double
#endif

void print_results(double *timings, int iter, int size, std::string benchmark, int dim, int bench);

void delay_time(int size);

#endif