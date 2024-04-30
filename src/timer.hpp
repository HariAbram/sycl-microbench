#include <iostream>
#include <math.h>
#include <numeric>
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