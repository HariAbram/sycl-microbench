#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <algorithm>

#include "../include/timer.hpp"
#include "../include/micro-bench-omp.hpp"
#include "../include/utils.hpp"
#include "../include/kernels.hpp"

#ifndef TYPE
#define TYPE double
#endif


void std_memory_alloc(int size, int iter, bool print)
{
  timer time;
  timer time1;

  auto timings_alloc = (double*)std::malloc(sizeof(double)*iter);
  
  for (size_t i = 0; i < iter; i++)
  {
      time.start_timer();
      volatile TYPE * m = (TYPE *)std::malloc(sizeof(TYPE)*size*size);
      m[size] = size;
      free((TYPE*)m);
      time.end_timer();
      timings_alloc[i] = time.duration();
      
  }
  
  if (print)
  {
      print_results(timings_alloc, iter, size, "Host memory alloc(ms)",1, 1);
  }
  
  auto timings = (double*)malloc(sizeof(double)*iter);

  TYPE* m = (TYPE *)std::aligned_alloc(sizeof(TYPE)*1024*1024,sizeof(TYPE)*size*size);
  TYPE* a = (TYPE *)std::aligned_alloc(sizeof(TYPE)*1024*1024,sizeof(TYPE)*size*size);

  std::fill(m,m+(size*size),0.0);
  std::fill(a,a+(size*size),1.0);

  for (size_t i = 0; i < iter; i++)
  {
    time1.start_timer();
    kernel_copy( m,  a, size);
    time1.end_timer();

    timings[i] = time1.duration();   
  }

  free(m);
  free(a);
  if (print)
  {
    print_results(timings,iter,size, "std memory", 1, 1);
  }
}

void parallel_for_omp(int size, bool print, int iter)
{
    
    timer time;

    TYPE * sum = (TYPE * )malloc(sizeof(TYPE)*size*size); 

    std::fill(sum, sum+size*size, 0.0);

    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
      time.start_timer();
      kernel_parallel_omp(size, sum);
      time.end_timer();

      timings[i] = time.duration();
    };

    if(sum[0] != 1024*iter) 
    {
      std::cout << "Verification failed "
                << "Expected value "<< 1024*iter
                << "Final value"<< sum[0]
                <<std::endl;
    }

    if (print)
    {
      print_results(timings, iter, size, "OMP_parallel", 1, 2);
    }
    
    free(sum);

}

void parallel_for_omp_nested(int size, bool print, int iter)
{
    
    timer time;

    TYPE * sum = (TYPE * )malloc(sizeof(TYPE)*size*size); 

    std::fill(sum, sum+size*size, 0.0);

    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    

    for ( i = 0; i < iter; i++)
    {
      time.start_timer();
      kernel_parallel_omp_nested(size, sum);
      time.end_timer();

      timings[i] = time.duration();

      if(sum[0] < 0)std::cout<<sum<<std::endl;

    };

    if(sum[0] != 1024*iter)
    {
      std::cout << "Verification failed "
                << "Expected value "<< 1024*iter
                << "Final value"<< sum[0]
                <<std::endl;
    }

    if (print)
    {
      print_results(timings, iter, size, "OMP_parallel_nested", 1, 2);
    }
    
    free(sum);

}

void atomics_omp( int size, bool print, int iter)
{
  timer time;

  TYPE * m = (TYPE * )malloc(sizeof(TYPE)*size*size); 
  std::fill(m , m+(size*size),1);

  int i;
  TYPE sum = 0.0;

  auto timings = (double*)std::malloc(sizeof(double)*iter);
  for ( i = 0; i < iter; i++)
  {

    time.start_timer();
    kernel_atomics(size, sum, m);
    time.end_timer();

    timings[i] = time.duration();

    if(sum < 0) std::cout<<sum<<std::endl;
  };

  if (sum!= size*size*iter)
  {
    std::cout << "Verification failed "
              << "Expected value "<< size*size*iter
              << "Final value"<< sum
              <<std::endl;
  }

  if (print)
  {
    print_results(timings, iter, size, "OMP atomics", 1, 3);
  }

  free(m);
}

void reduction_omp(int size, bool print, int iter)
{

  timer time;

  TYPE * m = (TYPE * )malloc(sizeof(TYPE)*size*size); 
  std::fill(m , m+(size*size),1);
  
  int i;
  TYPE sum = 0.0;

  auto timings = (double*)std::malloc(sizeof(double)*iter);

  for ( i = 0; i < iter; i++)
  {

    time.start_timer();
    kernel_reduction( size, sum, m);
    time.end_timer();

    timings[i] = time.duration();

    if(sum < 0) std::cout<<sum<<std::endl;
  };

  if (sum!= size*size*iter)
  {
    std::cout << "Verification failed "
              << "Expected value "<< size*size*iter
              << "Final value"<< sum
              <<std::endl;
  }

  if (print)
  {
    print_results(timings, iter, size, "OMP Reduction", 1, 3);
  }
  
  free(m);

}

void barrier_test_omp(int size, bool print, int iter )
{

  timer time;

  TYPE * sum = (TYPE * )malloc(sizeof(TYPE)*size*size); 

  std::fill(sum , sum+(size*size),0);

  int i;

  auto timings = (double*)std::malloc(sizeof(double)*iter); 

  for ( i = 0; i < iter; i++)
  {
    time.start_timer();
    kernel_barrier_omp(size, sum);
    time.end_timer();

    timings[i] = time.duration();
    
    if(sum[0] < 0) std::cout<<sum<<std::endl;
  };

  if(sum[0] != 1024*iter)
  {
    std::cout << "Verification failed "
              << "Expected value "<< 1024*iter
              << "Final value"<< sum[0]
              <<std::endl;
  }

  if (print)
  {
    print_results(timings, iter, size, "OMP barriers", 1, 4);
  }
  free(sum);
}