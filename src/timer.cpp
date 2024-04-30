#include <iostream>
#include <math.h>
#include <numeric>
#include <vector>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <iomanip>

#include "timer.hpp"

timer::timer()
{
}

timer::~timer()
{
}

void timer::start_timer()
{
    this->start_time = std::chrono::high_resolution_clock::now();
}

void timer::end_timer()
{
    this->end_time = std::chrono::high_resolution_clock::now();
}

double timer::duration()
{
    this->duration_timer = std::chrono::duration_cast<std::chrono::nanoseconds>(this->end_time  - this->start_time);
    return (double) this->duration_timer.count(); 
}

