#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <omp.h>

#include "timer.cpp"
#include "micro-bench-omp.hpp"


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
        std::cout<<"Usage: "<< argv[0]<< "[-s size <optional>|-b blocksize <optional>|-i iteration <optional>] \n" << std::endl;
        exit(EXIT_FAILURE);
        }
    }
    
   
    int tile_size = 16;

    parallel_for_omp(n_col, tile_size);

    parallel_for_omp_nested(n_col, tile_size);

    atomic_reduction_omp(n_col, tile_size);

    reduction_without_atomics_omp(n_col, tile_size);

    barrier_test_omp(n_col, tile_size);


    return 0;

}

void parallel_for_omp(int size, int block_size)
{
    
    timer time;

    double * m = (double * )malloc(sizeof(double)*size*size); 

    int i;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
      #pragma omp parallel for
      for (size_t j = 0; j < size*size; j++)        
      {
        double sum = 0;
        for (size_t l = 0; l < 1024; l++)
        {
            sum += 1;
        }
          
      };
    };
       

    time.end_timer();

    auto kernel_offload_time = time.duration();

    std::cout << "Total time taken for the execution of parallel for in omp "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    free(m);

}

void parallel_for_omp_nested(int size, int block_size)
{
    
    timer time;

    double * m = (double * )malloc(sizeof(double)*size*size); 

    int i;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
      #pragma omp parallel for collapse(2)
      for (size_t j = 0; j < size; j++)        
      {
        for (size_t k = 0; k < size; k++)
        {
            double sum = 0;
            for (size_t l = 0; l < 1024; l++)
            {
                sum += 1;
            }
        }
          
      };
    };
       

    time.end_timer();

    auto kernel_offload_time = time.duration();

    std::cout << "Total time taken for the execution of nested parallel for in omp "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    free(m);

}

void atomic_reduction_omp( int size, int block_size)
{
  timer time;

  double * m = (double * )malloc(sizeof(double)*size); 
  std::fill(m , m+size,1);
  

  int i;

  time.start_timer();

  for ( i = 0; i < 1024; i++)
  {
    double sum = 0.0;
    #pragma omp parallel for 
    for (size_t j = 0; j < size; j++)        
    {
      #pragma omp atomic
      sum+= m[j];

    };
  };
      
  
  time.end_timer();

  auto kernel_offload_time = time.duration();

  std::cout << "Total time taken for the execution of atomic construct in omp "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
  
  free(m);
}

void reduction_without_atomics_omp(int size, int tile_size)
{

  timer time;

  double * m = (double * )malloc(sizeof(double)*size); 
  std::fill(m , m+size,1);
  

  int i;

  time.start_timer();

  for ( i = 0; i < 1024; i++)
  {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t j = 0; j < size; j++)        
    {
      sum+= m[j];

    };
  };
      
  
  time.end_timer();

  auto kernel_offload_time = time.duration();

  std::cout << "Total time taken for the execution of reduction construct in omp "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
  
  free(m);

}

void barrier_test_omp(int size, int block_size)
{

  timer time;

  double * m = (double * )malloc(sizeof(double)*size); 

  int i;

  time.start_timer();

  for ( i = 0; i < 1024; i++)
  {
    #pragma omp parallel for 
    for (size_t j = 0; j < size; j++)        
    {

        double sum = 0;
        for (size_t l = 0; l < 1024; l++)
        {
            sum += 1;
        } 
        #pragma omp barrier        
    };
    
  };
      

  time.end_timer();

  auto kernel_offload_time = time.duration();

  std::cout << "Total time taken for the execution of barrier construct in omp "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
  
  free(m);

}