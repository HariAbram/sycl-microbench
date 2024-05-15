#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <sys/time.h>
#include <omp.h>

#ifndef TYPE
#define TYPE double
#endif


#include "timer.hpp"
#include "parallel-bench.hpp"
#include "vectorization-bench.hpp"


using namespace cl;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"block size", 1, NULL, 'b'},
  {"size", 1, NULL, 's'},
  {"vec-add", 0, NULL, 'v'},
  {"mat-mul", 0, NULL, 'm'},
  {"mem-alloc", 0, NULL, 'a'},
  {"reduction", 0, NULL, 'r'},
  {"range", 0, NULL, 'e'},
  {"ndrange", 0, NULL, 'n'},
  {"barrier", 0, NULL, 'w'},
  {"index_m", 1, NULL, 'i'},
  {"help", 0, NULL, 'h'},
  {0,0,0,0}
};

int main(int argc, char* argv[]) {

    int n_row, n_col;
    n_row = n_col = 32; // deafult matrix size
    int opt, option_index=0;
    int block_size = 16;


    bool vec_add = false;
    bool mat_mul=false;
    bool mem_alloc=false;
    bool reduction=false;
    bool range=false;
    bool nd_range=false;
    bool barrier=false;

    char * vec = NULL;
    int vec_no = 1;
    char *OMP_pragmas = NULL;

    bool help = false;


    while ((opt = getopt_long(argc, argv, ":s:b:v:i:h:m:r:a:e:n:w:", 
          long_options, &option_index)) != -1 ) {
    switch(opt){
      case 's':
        n_col=n_row= atoi(optarg);
        break;
      case 'b':
        block_size = atoi(optarg);
        break;
      case 'v':
        vec_add = true;
        break;
      case 'm':
        mat_mul = true;
        break;
      case 'r':
        reduction = true;
        break;
      case 'e':
        range = true;
        break;
      case 'n':
        nd_range = true;
        break;
      case 'a':
        mem_alloc = true;
        break;
      case 'w':
        barrier = true;
        break;
      case 'i':
        vec_no = atoi(optarg);
        break;
      case 'h':
        help = true;
        break;
      case '?':
        fprintf(stderr, "invalid option\n");
        break;
      case ':':
        fprintf(stderr, "missing argument\n");
        break;
      default:
        fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
            argv[0]);
        exit(EXIT_FAILURE);
        }
    }

    if ( (optind < argc) || (optind == 1)) {
      fprintf(stderr, "No input parameters specified, use --help to see how to use this binary\n");
      exit(EXIT_FAILURE);
    } 

    if (help)
    {

      std::cout<<"Usage: \n"<< argv[0]<< " [-s size |-b blocksize <optional>|\n"
                                        " --mat-mul : to run matrix multiplication \n" 
                                        " --vec-add : to run vector addition \n"
                                        "             can run only mat-mul or vec-add at a time, can't run both simultaneously \n"
                                        " --mem-alloc : to alloc memory using SYCL and standard malloc \n"
                                        " --reduction : to test reduction using atomics and sycl reduction construct"
                                        " --range : to test sycl range construct"
                                        " --ndrange : to test sycl nd_range construct"
                                        " --barrier : to test sycl barrier construct"
                                        " -i : for different routines in vectorization benchmark\n"
                                        "       1 - range with USM\n"
                                        "       2 - range with Buffer and Accessors\n"
                                        "       3 - nd_range with USM\n"
                                        "       4 - nd_range with Buffer and Accessor\n"<< std::endl;
      
      exit(EXIT_FAILURE);
    }
    

    
    sycl::queue Q{};
    std::cout << "running on ..."<< std::endl;
    std::cout << Q.get_device().get_info<sycl::info::device::name>()<<"\n"<<std::endl;

    if (mat_mul)
    {
      if (vec_no==1)
      {
        mat_mul_range_usm(Q, n_row);
      }
      else if (vec_no == 2)
      {
        mat_mul_range_buff_acc(Q, n_row);
      }
      else if (vec_no == 3)
      {
        mat_mul_ndrange_usm(Q, n_row, block_size);
      }
      else if (vec_no == 4)
      {
        mat_mul_ndrange_buff_acc(Q, n_row, block_size);
      }

    }
    else if (vec_add)
    {
      if (vec_no==1)
      {
        vec_add_range_usm(Q, n_row);
      }
      else if (vec_no == 2)
      {
        vec_add_range_buff_acc(Q, n_row);
      }
      else if (vec_no == 3)
      {
        vec_add_ndrange_usm(Q, n_row, block_size);
      }
      else if (vec_no == 4)
      {
        vec_add_ndrange_buff_acc(Q, n_row, block_size);
      }

    }
    else if (mem_alloc)
    {

      host_memory_alloc(Q, n_row, false);

      host_memory_alloc(Q, n_row, true);

      shared_memory_alloc(Q, n_row,false);

      shared_memory_alloc(Q, n_row,true);

      device_memory_alloc(Q, n_row,false);

      device_memory_alloc(Q, n_row,true);

      timer time;
      timer time1;

      float time_fill = 0;

      struct timeval tv;

      time.start_timer();
      for (size_t i = 0; i < 10; i++)
      {
          volatile TYPE * m = (TYPE *)std::malloc(sizeof(TYPE)*n_row*n_row);
          m[n_row] = n_row;
          free((TYPE*)m);
          
      }
      time.end_timer();
 

      auto kernel_offload_time = time.duration();

      for (size_t i = 0; i < 10; i++)
      {
          TYPE* m = (TYPE *)std::malloc(sizeof(TYPE)*n_row*n_row);
          TYPE* a = (TYPE *)std::malloc(sizeof(TYPE)*n_row*n_row);

          std::fill(a,a+(n_row*n_row),1);

          time1.start_timer();
          #pragma omp parallel for 
          for (size_t j = 0; j < n_row*n_row; j++)
          {
            m[j] = a[1];
          }
          
          time1.end_timer();
          time_fill += time1.duration();
          free(m);
      }

      std::cout << "Total time taken for the memory allocation with malloc "<< kernel_offload_time/10 << " nanoseconds\n" 
                   "fill time is "<< time_fill/(10*1E3)<< " microseconds"<<std::endl;



    }

    else if (reduction)
    {
      reduction_with_atomics_buf_acc(Q, n_row, false);

      reduction_with_atomics_buf_acc(Q, n_row, true);

      reduction_with_atomics_usm(Q, n_row, false);

      reduction_with_atomics_usm(Q, n_row, true);
      
      reduction_with_buf_acc(Q, n_row,  block_size, false);

      reduction_with_buf_acc(Q, n_row,  block_size, true);
    }
    else if (range)
    {
      range_with_usm(Q, n_row, 1,false);

      range_with_usm(Q, n_row, 1,true);

      range_with_usm(Q, n_row, 2,false);

      range_with_usm(Q, n_row, 2,true);

      range_with_buff_acc(Q, n_row ,1,false);

      range_with_buff_acc(Q, n_row ,1,true);
      
      range_with_buff_acc(Q, n_row ,2,false);

      range_with_buff_acc(Q, n_row ,2,true);

      
    }
    else if (nd_range)
    {
      nd_range_with_usm(Q, n_row, block_size ,1, false);

      nd_range_with_usm(Q, n_row, block_size ,1, true);

      nd_range_with_usm(Q, n_row, block_size ,2, false);

      nd_range_with_usm(Q, n_row, block_size ,2, true);

      nd_range_with_buff_acc(Q, n_row, block_size ,1, false);
      
      nd_range_with_buff_acc(Q, n_row, block_size ,1, true);

      nd_range_with_buff_acc(Q, n_row, block_size ,2, false);

      nd_range_with_buff_acc(Q, n_row, block_size ,2, true);
    }
    
    else if (barrier)
    {

      global_barrier_test_usm(Q, n_row, block_size, false);

      global_barrier_test_usm(Q, n_row, block_size, true);

      global_barrier_test_buff_acc(Q, n_row,  block_size, false);

      global_barrier_test_buff_acc(Q, n_row,  block_size, true);

      local_barrier_test_usm(Q, n_row, block_size, false);

      local_barrier_test_usm(Q, n_row, block_size, true);

      local_barrier_test_buff_acc(Q, n_row, block_size, false);

      local_barrier_test_buff_acc(Q, n_row, block_size, true);

    }

    else
    {
      fprintf(stderr, "No input parameters specified, use --help to see how to use this binary\n"); 
    }


    return 0;

}






