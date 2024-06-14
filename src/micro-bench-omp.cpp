#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <omp.h>
#include <algorithm>

#include "timer.hpp"
#include "micro-bench-omp.hpp"

#ifndef TYPE
#define TYPE double
#endif

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

      #pragma omp parallel 
      {
        #pragma omp for 
        for (size_t j = 0; j < size*size; j++)        
        {
          
          for (size_t l = 0; l < 1024; l++)
          {
              sum[j] += 1;
              
          }
            
        };

      }

      time.end_timer();

      timings[i] = time.duration();
    };
       
    auto minmax = std::minmax_element(timings, timings+iter);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    if(sum[0] != 1024*iter) 
    {
      std::cout << "Verification failed "
                << "Expected value "<< 1024*iter
                << "Final value"<< sum[0]
                <<std::endl;
    }
    

    if (print)
    {
        std::cout
            << std::left << std::setw(24) << "OMP_parallel"
            << std::left << std::setw(24) << 1
            << std::left << std::setw(24) << *minmax.first*1E-9
            << std::left << std::setw(24) << *minmax.second*1E-9
            << std::left << std::setw(24) << average*1E-9
            << std::endl
            << std::fixed;
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

      #pragma omp parallel  
      {
        #pragma omp for collapse(2)
        for (size_t j = 0; j < size; j++)        
        {
          for (size_t k = 0; k < size; k++)
          {
              for (size_t l = 0; l < 1024; l++)
              {
                  sum[j*size+k] += 1;
              }
          }
        };
      }

      time.end_timer();

      timings[i] = time.duration();

      if(sum[0] < 0)std::cout<<sum<<std::endl;

    };

    auto minmax = std::minmax_element(timings, timings+iter);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    if(sum[0] != 1024*iter)
    {
      std::cout << "Verification failed "
                << "Expected value "<< 1024*iter
                << "Final value"<< sum[0]
                <<std::endl;
    }

    if (print)
    {
        std::cout
            << std::left << std::setw(24) << "OMP_parallel_nested"
            << std::left << std::setw(24) << 1
            << std::left << std::setw(24) << *minmax.first*1E-9
            << std::left << std::setw(24) << *minmax.second*1E-9
            << std::left << std::setw(24) << average*1E-9
            << std::endl
            << std::fixed;
    }
    
    free(sum);

}

void atomic_reduction_omp( int size, bool print, int iter)
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
    
    #pragma omp parallel for private(sum)
    for (size_t j = 0; j < size*size; j++)        
    {
      #pragma omp atomic
      sum+= m[j];

    };

    time.end_timer();

    timings[i] = time.duration();

    if(sum < 0) std::cout<<sum<<std::endl;
  };

  auto minmax = std::minmax_element(timings, timings+iter);

  double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

  if (sum!= size*size*iter)
  {
    std::cout << "Verification failed "
              << "Expected value "<< size*size*iter
              << "Final value"<< sum
              <<std::endl;
  }

  if (print)
  {
      std::cout
          << std::left << std::setw(24) << "OMP_atomics"
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl
          << std::fixed;
  }

  free(m);
}

void reduction_without_atomics_omp(int size, bool print, int iter)
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

    #pragma omp parallel for reduction(+:sum) 
    for (size_t j = 0; j < size*size; j++)        
    {
      sum+= m[j];

    };

    time.end_timer();

    timings[i] = time.duration();

    if(sum < 0) std::cout<<sum<<std::endl;
  };

  auto minmax = std::minmax_element(timings, timings+iter);

  double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

  if (sum!= size*size*iter)
  {
    std::cout << "Verification failed "
              << "Expected value "<< size*size*iter
              << "Final value"<< sum
              <<std::endl;
  }

  if (print)
  {
      std::cout
          << std::left << std::setw(24) << "OMP_reduction"
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl
          << std::fixed;
  }
  
  free(m);

}

void barrier_test_omp(int size, bool print, int iter)
{

  timer time;

  TYPE * sum = (TYPE * )malloc(sizeof(TYPE)*size*size); 

  std::fill(sum , sum+(size*size),0);

  int i;

  auto timings = (double*)std::malloc(sizeof(double)*iter); 

  for ( i = 0; i < iter; i++)
  {
    time.start_timer();

    #pragma omp parallel 
    {
      #pragma omp for
      for (size_t j = 0; j < size*size; j++)        
      {
          
          for (size_t l = 0; l < 1024; l++)
          {
              sum[j] += 1;
          } 
                  
      };

      #pragma omp barrier
    }

    time.end_timer();

    timings[i] = time.duration();
    
    if(sum[0] < 0) std::cout<<sum<<std::endl;
  };

  auto minmax = std::minmax_element(timings, timings+iter);

  double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

  if(sum[0] != 1024*iter)
  {
    std::cout << "Verification failed "
              << "Expected value "<< 1024*iter
              << "Final value"<< sum[0]
              <<std::endl;
  }

  if (print)
  {
      std::cout
          << std::left << std::setw(24) << "OMP_barriers"
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl
          << std::fixed;
  }
  
  free(sum);

}

void kernel_computation()
{
  int sum = 0;

  for (size_t l = 0; l < 1024; l++)
  {
      sum += 1;
      if(sum < 0) std::cout<<sum<<std::endl;
  }
}