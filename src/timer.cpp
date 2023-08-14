#include <iostream>
#include <math.h>
#include <numeric>
#include <execution>
#include <vector>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <iomanip>

class timer
{
private:
    std::chrono::_V2::system_clock::time_point start_time ;
    std::chrono::_V2::system_clock::time_point end_time ;
    std::chrono::nanoseconds duration_timer;

public:
    timer();
    ~timer();
    void start_timer();
    void end_timer();
    double duration();
};

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

