#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>

#include "utils.hpp"
#include "timer.cpp"
#include "micro-bench.hpp"

using namespace cl;

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
    func_ret_t ret, ret1;
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
    

    sycl::queue Q{sycl::cpu_selector{}};
    std::cout << "running on ..."<< std::endl;
    std::cout << Q.get_device().get_info<sycl::info::device::name>()<<"\n"<<std::endl;


    host_memory_alloc(Q, n_row);

    shared_memory_alloc(Q, n_row);

    device_memory_alloc(Q, n_row);

//    kernel_offloading_duration(Q, n_row);

    range_1d_with_usm(Q, n_row, block_size);
    
    range_2d_with_usm(Q, n_row, block_size);
    
    ndrange_1d_with_usm(Q, n_row, block_size);

    ndrange_2d_with_usm(Q, n_row, block_size);

    atomic_test_with_usm_and_range_1dim(Q, n_row, block_size);

    int tile_size = 16;

    reduction_without_atomics(Q, n_row, tile_size);

    barrier_test(Q, n_row, block_size);

    local_barrier_test(Q, n_row, block_size);

    range_1d_with_buff_acc(Q,  n_row,  block_size);

    range_2d_with_buff_acc(Q,  n_row,  block_size);

    ndrange_1d_with_buff_acc(Q,  n_row,  block_size);

    ndrange_2d_with_buff_acc(Q,  n_row,  block_size);

    auto delay = dealy_time();
    std::cout << "delay time is: "<< delay << " seconds" << std::endl;

    return 0;

}



// micro bench functions

double dealy_time()
{
    timer time;
    time.start_timer();
    double sum = 0;
    for (size_t l = 0; l < 1024; l++)
    {
        sum += 1;
    }
    time.end_timer();
    auto kernel_offload_time = time.duration()/(1E+9);

    return kernel_offload_time;

}

void host_memory_alloc(sycl::queue &Q, int size)
{

    timer time;
    time.start_timer();
    int i;

    for ( i = 0; i < 1024; i++)
    {
        auto m_host = sycl::malloc_host<double>(size*size*sizeof(double),Q); Q.wait();
        free(m_host,Q);
    }
    
    time.end_timer();

    auto host_alloc_time = time.duration();
    std::cout << "Total time taken for the host memory allocation with sycl "<< host_alloc_time/1024 << " nanoseconds\n" << std::endl;

}

void shared_memory_alloc(sycl::queue &Q, int size)
{

    timer time;
    time.start_timer();
    int i;

    for ( i = 0; i < 1024; i++)
    {
        auto m_shared = sycl::malloc_shared<double>(size*size*sizeof(double),Q); Q.wait();
        free(m_shared,Q);
    }
    
    time.end_timer();

    auto shared_alloc_time = time.duration();
    std::cout << "Total time taken for the shared memory allocation with sycl "<< shared_alloc_time/1024 << " nanoseconds\n" << std::endl;

}

void device_memory_alloc(sycl::queue &Q, int size)
{

    timer time;
    time.start_timer();
    int i;

    for ( i = 0; i < 1024; i++)
    {
        auto m_device = sycl::malloc_device<double>(size*size*sizeof(double),Q); Q.wait();
        free(m_device,Q);
    }
    
    time.end_timer();

    auto device_alloc_time = time.duration();
    std::cout << "Total time taken for the device memory allocation with sycl "<< device_alloc_time/1024 << " nanoseconds\n" << std::endl;

}

void kernel_offloading_duration(sycl::queue &Q, int size)
{
    timer time;

    double * m1 = (double *)malloc(sizeof(double)*size*size);

    //const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};

    sycl::buffer<double , 1> d_m1(m1,size*size);

    time.start_timer();
    int i;

    for ( i = 0; i < 1024; i++)
    {
        Q.submit([&](sycl::handler& cgh){
            auto m1_acc = d_m1.get_access<sycl::access::mode::read>(cgh);

        });
        Q.wait();
    }

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the kernel offloading and invoking accessor "<< kernel_offload_time/1024 << " nanoseconds\n" << std::endl;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.submit([&](sycl::handler& cgh){

        });
        Q.wait();
    }

    time.end_timer();

    kernel_offload_time = time.duration();
    std::cout << "Total time taken for the just kernel offloading "<< kernel_offload_time/1024 << " nanoseconds\n" << std::endl;
    
    
}

void range_1d_with_usm(sycl::queue &Q, int size, int block_size)
{

    timer time;
    
    auto N = static_cast<size_t>(size*size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    int i;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.parallel_for<class range_1d_usm>(sycl::range<1>(global), [=](sycl::item<1>it){

            double sum = 0;
            for (size_t l = 0; l < 1024; l++)
            {
                sum += 1;
            }
            
            
        });
        Q.wait();
    }
    

    

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the execution of range parallel construct with 1 dim "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    
}

void range_2d_with_usm(sycl::queue &Q, int size, int block_size)
{

    timer time;
    
    auto N = static_cast<size_t>(size);
    sycl::range<2> global{N,N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<2> local{N_b,N_b};

    int i;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.parallel_for<class range_2d_usm>(sycl::range<2>(global), [=](sycl::item<2>it){

            double sum = 0;
            for (size_t l = 0; l < 1024; l++)
            {
                sum += 1;
            }
        
        });
        Q.wait();
    }
    

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the execution of range parallel construct with 2 dim "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    
}

void ndrange_1d_with_usm(sycl::queue &Q, int size, int block_size)
{

    timer time;
    
    auto N = static_cast<size_t>(size*size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    int i;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.parallel_for<class ndrange_1d_usm>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            double sum = 0;
            for (size_t l = 0; l < 1024; l++)
            {
                sum += 1;
            }
            
        });
        Q.wait();
    }
      

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the execution of nd_range parallel construct with 1 dim "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    
}

void ndrange_2d_with_usm(sycl::queue &Q, int size, int block_size)
{

    timer time;
    
    auto N = static_cast<size_t>(size);
    sycl::range<2> global{N,N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<2> local{N_b,N_b};

    int i;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.parallel_for<class ndrange_2d_usm>(sycl::nd_range<2>(global,local), [=](sycl::nd_item<2>it){

            double sum = 0;
            for (size_t l = 0; l < 1024; l++)
            {
                sum += 1;
            }
        
        });
        Q.wait();
    }
    

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the execution of nd_range parallel construct with 2 dim "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    
}

void atomic_test_with_usm_and_range_1dim(sycl::queue &Q, int size, int block_size)
{
    timer time;

    auto m_shared = sycl::malloc_shared<double>(size*sizeof(double),Q); Q.wait();
    std::fill(m_shared,m_shared+size,1.0);
    auto sum = sycl::malloc_shared<double>(1*sizeof(double),Q); Q.wait();

    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    int i;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.parallel_for<class reduction_atomic>(sycl::range<1>(global), [=](sycl::item<1>it){

            auto j = it.get_id(0);

            auto v = sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(
            sum[0]);

            
            v.fetch_add(m_shared[j]);
            
        });

        Q.wait();
    }   

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the execution of atomic ref construct with range parallel for in 1 dim  "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    
    free(m_shared,Q);
}


void reduction_without_atomics(sycl::queue &Q, int size, int tile_size)
{
    
    timer time;

    auto m_shared = sycl::malloc_shared<double>(size*sizeof(double),Q); Q.wait();
    auto accum_shared = sycl::malloc_shared<double>(size/tile_size*sizeof(double),Q); Q.wait();
    auto sum = 0.0;

    auto N = static_cast<size_t>(size/tile_size);
    sycl::range<1> global{N};
    auto tile = static_cast<size_t>(tile_size);
 
    int i;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.parallel_for<class reduction_notatomic>(sycl::range<1>(global), [=](sycl::item<1>it){

            auto j = it.get_id(0);

            for (size_t k = 0; k < tile; k++)
            {
                accum_shared[k] += m_shared[j*tile + k];
            }
            
            
        });

        Q.wait();

        std::accumulate(accum_shared, accum_shared+(size/tile_size), sum);
    }   

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the execution of reduction without atomics  "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    
    free(m_shared,Q);

}


void barrier_test(sycl::queue &Q, int size, int block_size)
{

    timer time;

    
    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    int i;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.parallel_for<class global_barrier>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            double sum = 0;
            for (size_t l = 0; l < 1024; l++)
            {
                sum += 1;
            }
            it.barrier();
        
        });
        Q.wait();
    }
    

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the global barrier scope with 1 dim "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    
}

void local_barrier_test(sycl::queue &Q, int size, int block_size)
{
    timer time;

    
    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    int i;

    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.parallel_for<class local_barrier>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            double sum = 0;
            for (size_t l = 0; l < 1024; l++)
            {
                sum += 1;
            }
            it.barrier(sycl::access::fence_space::local_space);
        
        });
        Q.wait();
    }
    

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the local barrier scope with 1 dim "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    

}


void range_1d_with_buff_acc(sycl::queue &Q, int size, int block_size)
{
    timer time;

                                                                                                                                                                                                                                             
    auto N = static_cast<size_t>(size*size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    int i;


    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.submit([&](sycl::handler& cgh){

            cgh.parallel_for<class range_1d_BandA>(sycl::range<1>(global), [=](sycl::item<1>it){

                double sum = 0;
                for (size_t l = 0; l < 1024; l++)
                {
                    sum += 1;
                }
                
            });

        });

        
        Q.wait();

    }    

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the execution of range parallel construct with 1 dim \n and with buffer and accessors "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    
}




void range_2d_with_buff_acc(sycl::queue &Q, int size, int block_size)
{
    timer time;

                                                                                                                                                                                                                                             
    auto N = static_cast<size_t>(size);
    sycl::range<2> global{N,N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<2> local{N_b, N_b};

    int i;


    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.submit([&](sycl::handler& cgh){

            cgh.parallel_for<class range_2d_BandA>(sycl::range<2>(global), [=](sycl::item<2>it){

                double sum = 0;
                for (size_t l = 0; l < 1024; l++)
                {
                    sum += 1;
                }
                
            });

        });


        Q.wait();
        
    }
    

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the execution of range parallel construct with 2 dim \n and with buffer and accessors "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    
}

void ndrange_1d_with_buff_acc(sycl::queue &Q, int size, int block_size)
{
    timer time;

                                                                                                                                                                                                                                             
    auto N = static_cast<size_t>(size*size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    int i;


    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.submit([&](sycl::handler& cgh){

            cgh.parallel_for<class ndrange_1d_BandA>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

                double sum = 0;
                for (size_t l = 0; l < 1024; l++)
                {
                    sum += 1;
                }
                
            });

        });
        Q.wait();
        
    }
    

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the execution of nd_range parallel construct with 1 dim \n and with buffer and accessors "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    
}


void ndrange_2d_with_buff_acc(sycl::queue &Q, int size, int block_size)
{
    timer time;

                                                                                                                                                                                                                                             
    auto N = static_cast<size_t>(size);
    sycl::range<2> global{N,N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<2> local{N_b, N_b};

    int i;


    time.start_timer();

    for ( i = 0; i < 1024; i++)
    {
        Q.submit([&](sycl::handler& cgh){

            cgh.parallel_for<class ndrange_2d_BandA>(sycl::nd_range<2>(global,local), [=](sycl::nd_item<2>it){

                double sum = 0;
                for (size_t l = 0; l < 1024; l++)
                {
                    sum += 1;
                }
                
            });

        });

        Q.wait();
        
    }
    

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the execution of nd_range parallel construct with 2 dim \n and with buffer and accessors "<< kernel_offload_time/(1024*1E9) << " seconds\n" << std::endl;
    
    

}