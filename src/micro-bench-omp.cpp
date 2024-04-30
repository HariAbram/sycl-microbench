#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <omp.h>

#include "timer.hpp"
#include "micro-bench-omp.hpp"

#ifndef TYPE
#define TYPE double
#endif

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"block size", 1, NULL, 'b'},
  {"size", 1, NULL, 's'},
  {0,0,0,0}
};

int main(int argc, char* argv[]) {

    int n_row, n_col;
    n_row = n_col = 1024; // deafult matrix size
    int opt, option_index=0;
    int block_size = 16;
    float *  __restrict__ m1,* __restrict__ m2;
    int iterations = 5;
    char * type;


    while ((opt = getopt_long(argc, argv, "::s:b:", 
          long_options, &option_index)) != -1 ) {
    switch(opt){
      case 'b':
        block_size = atoi(optarg);
        break;
      case 's':
        n_col=n_row= atoi(optarg);
        break;
      case '?':
        fprintf(stderr, "invalid option\n");
        break;
      case ':':
        fprintf(stderr, "missing argument\n");
        break;
      default:
        std::cout<<"Usage: "<< argv[0]<< "[-s size |-b blocksize <optional>] \n" << std::endl;
        exit(EXIT_FAILURE);
        }
    }
    
   
    int tile_size = 16;

    parallel_for_omp(n_col, tile_size);

    parallel_for_omp(n_col, tile_size);

    parallel_for_omp_nested(n_col, tile_size);

    parallel_for_omp_nested(n_col, tile_size);

    atomic_reduction_omp(n_col, tile_size);

    atomic_reduction_omp(n_col, tile_size);

    reduction_without_atomics_omp(n_col, tile_size);

    reduction_without_atomics_omp(n_col, tile_size);

    barrier_test_omp(n_col, tile_size);

    barrier_test_omp(n_col, tile_size);
    
    kernel_computation();


    return 0;

}

void parallel_for_omp(int size, int block_size)
{
    
    timer time;

    TYPE * sum = (TYPE * )malloc(sizeof(TYPE)*size*size); 

    std::fill(sum, sum+size*size, 0.0);

    int i;

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {
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

      
    };
       

    time.end_timer();

    if(sum[0] != 1024*10) std::cout<<"verification failed"<<std::endl;
    

    auto kernel_offload_time = time.duration();

    std::cout << "Total time taken for the execution of parallel for in omp "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    
    free(sum);

}

void parallel_for_omp_nested(int size, int block_size)
{
    
    timer time;

    TYPE * sum = (TYPE * )malloc(sizeof(TYPE)*size*size); 

    std::fill(sum, sum+size*size, 0.0);

    int i;

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {
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
      if(sum[0] < 0)std::cout<<sum<<std::endl;

    };
       

    time.end_timer();

    if(sum[0] != 1024*10) std::cout<<"verification failed"<<std::endl;


    auto kernel_offload_time = time.duration();

    std::cout << "Total time taken for the execution of nested parallel for in omp "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    
    free(sum);

}

void atomic_reduction_omp( int size, int block_size)
{
  timer time;

  TYPE * m = (TYPE * )malloc(sizeof(TYPE)*size*size); 
  std::fill(m , m+(size*size),1);
  

  int i;
  TYPE sum = 0.0;

  time.start_timer();

  for ( i = 0; i < 10; i++)
  {
    
    #pragma omp parallel for private(sum)
    for (size_t j = 0; j < size*size; j++)        
    {
      #pragma omp atomic
      sum+= m[j];

    };

    if(sum < 0) std::cout<<sum<<std::endl;
  };
      
  
  time.end_timer();

  


  auto kernel_offload_time = time.duration();

  std::cout << "Total time taken for the execution of atomic construct in omp "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
  
  free(m);
}

void reduction_without_atomics_omp(int size, int tile_size)
{

  timer time;

  TYPE * m = (TYPE * )malloc(sizeof(TYPE)*size*size); 
  std::fill(m , m+(size*size),1);
  

  int i;
  TYPE sum = 0.0;

  time.start_timer();

  for ( i = 0; i < 10; i++)
  {
     
    #pragma omp parallel for reduction(+:sum) 
    for (size_t j = 0; j < size*size; j++)        
    {
      sum+= m[j];

    };

    if(sum < 0) std::cout<<sum<<std::endl;
  };
      
  
  time.end_timer();

  


  auto kernel_offload_time = time.duration();

  std::cout << "Total time taken for the execution of reduction construct in omp "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
  
  free(m);

}

void barrier_test_omp(int size, int block_size)
{

  timer time;

  TYPE * sum = (TYPE * )malloc(sizeof(TYPE)*size*size); 

  std::fill(sum , sum+(size*size),0);

  int i;

  time.start_timer();

  for ( i = 0; i < 10; i++)
  {
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
    

    if(sum[0] < 0) std::cout<<sum<<std::endl;
  };
      

  time.end_timer();

  if(sum[0] != 1024*10) std::cout<<"verification failed "<< sum[0] <<std::endl;

  auto kernel_offload_time = time.duration();

  std::cout << "Total time taken for the execution of barrier construct in omp "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
  
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