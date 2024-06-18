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

#ifndef TYPE
#define TYPE double
#endif

#include "../include/timer.hpp"
#include "../include/parallel-bench.hpp"
#include "../include/kernels.hpp"
#include "../include/utils.hpp"


using namespace cl;


////////////////////////////////////////////////////////////////////////////////////////////////
// memory allocations

void memory_alloc(sycl::queue &Q, int size, int block_size , bool print, int iter)
{
    timer time;

    int i;

    auto N = static_cast<size_t>(size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);
    sycl::range<1> global{N*N};

    for (i = 0; i < iter; i++)
    {
        sycl::buffer<TYPE , 1> m_buff(global);
        sycl::buffer<TYPE , 1> a_buff(global);

        init_arrays(Q, m_buff, a_buff, global);

        time.start_timer();
        kernel_copy(Q, m_buff, a_buff, global);
        time.end_timer();

        timings[i] = time.duration();
    }

    if (print)
    {
        print_results(timings, iter, size, "memory B&A", 1, 1);
    }

    free(timings);

}

// sycl::range constuct

void range_with_buff_acc(sycl::queue &Q, int size, int dim, bool print, int iter)
{

    /*
    * creates a SYCL parallel region using <range> contruct for a given problem size. 
    * the dimensions of the range contruct can also be specified, the parameter <dim> 
    * takes values 1 or 2. 
    * 
    * This benchmark tests the overhead incurred for the thread creation. each thread 
    * computes a small kernel, which corresponds to dealy time. This benchmark uses 
    * buffer and accessors for memory management to store the variables. 
    * 
    */

    timer time;

    TYPE * sum = (TYPE *)std::malloc(size*size*sizeof(TYPE)); 

    std::fill(sum,sum+(size*size),0);
    
    auto N = static_cast<size_t>(size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        sycl::buffer<TYPE , 1> sum_buff((TYPE*)sum,size*size);

        int i;

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();

            kernel_parallel_1(Q, sum_buff, global);

            time.end_timer();

            timings[i] = time.duration();
        }

        auto sum_r = sum_buff.get_host_access();

        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "range_BA", 1, 2);
        }        
    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        sycl::buffer<TYPE , 1> sum_buff((TYPE*)sum,size*size);

        int i;

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_parallel_2(Q, sum_buff, global);
            time.end_timer();

            timings[i] = time.duration();
        }

        auto sum_r = sum_buff.get_host_access();

        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "range_BA", 2, 2);
        }
    } 
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }

    free((TYPE*)sum);
         
}


// sycl::nd_range constuct

void nd_range_with_buff_acc(sycl::queue &Q, int size, int block_size ,int dim, bool print, int iter)
{

    /*
    * creates a SYCL parallel region using <range> contruct for a given problem size 
    * the dimensions of the range contruct can also be specified, the parameter <dim> 
    * takes values 1 or 2. 
    * 
    * This benchmark tests the overhead incurred for the thread creation. each thread 
    * computes a small kernel, which corresponds to dealy time. This benchmark uses 
    * buffer and accessors for memory management to store the variables. 
    * 
    */

    timer time;

    TYPE * sum = (TYPE *)malloc(size*size*sizeof(TYPE)); 

    std::fill(sum,sum+(size*size),0);
   
    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};    

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        int i;

        auto N_b = static_cast<size_t>(block_size);
        if (block_size > size)
        {
            std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
            N_b = N;
        }
        sycl::range<1> local{N_b};

        sycl::buffer<TYPE , 1> sum_buff(sum,size*size);        

        for ( i = 0; i < iter; i++)
        {

            time.start_timer();
            kernel_parallel_1(Q, sum_buff, global, local);
            time.end_timer();

            timings[i] = time.duration();
        }

        auto sum_r = sum_buff.get_host_access();

        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<  std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "ndrange_BA", 1, 2);
        }
    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        int i;

        auto N_b = static_cast<size_t>(block_size);
        if (block_size > size)
        {
            std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
            N_b = N;
        }
        sycl::range<2> local{N_b,N_b};

        sycl::buffer<TYPE , 1> sum_buff(sum,size*size);

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();
            kernel_parallel_2(Q, sum_buff,global, local);
            time.end_timer();

            timings[i] = time.duration();
        }

        auto sum_r = sum_buff.get_host_access();

        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<std::endl;
        }

        if (print)
        {
            print_results(timings, iter, size, "ndrange_BA", 2, 2);
        }
    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }

    free(sum);
    
}


// reduction 

void reduction_with_atomics_buf_acc(sycl::queue &Q, int size, bool print, int iter)
{
    timer time;

    size = size*size;

    auto m = (TYPE *)std::malloc(size*sizeof(TYPE)); 
    std::fill(m,m+size,1.0);
    auto sum = (TYPE *)std::malloc(1*sizeof(TYPE)); 

    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};

    sycl::buffer<TYPE , 1> m_buff(m,size);

    sycl::buffer<TYPE , 1> sum_buff(sum,1);
    
    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();

        Q.submit([&](sycl::handler& cgh){
            auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);
            auto m_acc = m_buff.get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

                auto j = it.get_id(0);

                auto v = sycl::atomic_ref<TYPE, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space>(
                sum_acc[0]);

                
                v.fetch_add(m_acc[j]);
                
            });
        });

        Q.wait();

        time.end_timer();

        timings[i] = time.duration();
    }   

    auto minmax = std::minmax_element(timings, timings+iter);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);
    
    auto sum_r = sum_buff.get_host_access();


    if (sum_r[0]!= size*iter)
    {
        std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[0]
                      <<std::endl;
    }

    if (print)
    {
        std::cout
            << std::left << std::setw(24) << "atomics_BA"
            << std::left << std::setw(24) << *minmax.first*1E-9
            << std::left << std::setw(24) << *minmax.second*1E-9
            << std::left << std::setw(24) << average*1E-9
            << std::endl
            << std::fixed;
    }
    
    
    free(m);
    free(sum);
}

#if defined(REDUCTION_IN_SYCL) 

void reduction_with_buf_acc(sycl::queue &Q, int size, int block_size, bool print, int iter)
{
    timer time;

    auto m_shared = (TYPE *)std::malloc(size*sizeof(TYPE));
    std::fill(m_shared,m_shared+size,1.0);
    auto sum = sycl::malloc_shared<TYPE>(1*sizeof(TYPE),Q); Q.wait();

    auto N = static_cast<size_t>(size);
    auto N_b = static_cast<size_t>(block_size);

    sycl::range<1> global{N};
    sycl::range<1> local{N_b};

    sycl::buffer<TYPE , 1> m_buff(m_shared,size);

    sycl::buffer<TYPE , 1> sum_buff(sum,1);
        
    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();

        Q.submit([&](sycl::handler& cgh){
            
            auto m_acc = m_buff.get_access<sycl::access::mode::read>(cgh);    

#if defined(DPCPP) 
            auto sum_red = sycl::reduction(sum_buff, cgh,sycl::plus<TYPE>());
#else
            auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);
            auto sum_red = sycl::reduction(sum_acc, sycl::plus<TYPE>());
#endif

            cgh.parallel_for<>(sycl::nd_range<1>(global,local), sum_red ,[=](sycl::nd_item<1>it, auto &sum){

                auto j = it.get_global_id(0);

                sum += m_acc[j];
                
            });
        });

        Q.wait();

        time.end_timer();

        timings[i] = time.duration();
    }   

    auto minmax = std::minmax_element(timings, timings+iter);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
            << std::left << std::setw(24) << "Reduction"
            << std::left << std::setw(24) << *minmax.first*1E-9
            << std::left << std::setw(24) << *minmax.second*1E-9
            << std::left << std::setw(24) << average*1E-9
            << std::endl
            << std::fixed;
    }   
    
    free(m_shared);
}

#endif


void global_barrier_test_buff_acc(sycl::queue &Q, int size, int block_size, bool print, int iter)
{

    timer time;

    size = size*size;

    TYPE * sum = (TYPE *)malloc(size*sizeof(TYPE)); 

    std::fill(sum,sum+(size),0);

    
    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    sycl::buffer<TYPE , 1> sum_buff(sum,size);

    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();

        Q.submit([&](sycl::handler& cgh){
            auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class global_barrier_ba>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

                auto k = it.get_global_id(0);

                for (size_t l = 0; l < 1024; l++)
                {
                    sum_acc[k]+= 1;
                }

                
                it.barrier();
            
            });

        });
        
        Q.wait();

        time.end_timer();

        timings[i] = time.duration();
    }

    auto minmax = std::minmax_element(timings, timings+iter);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    auto sum_r = sum_buff.get_host_access();

    if (sum_r[0]!= 1024*iter)
    {
        std::cout << "Verification failed "
                  << "Expected value "<< 1024*iter
                  << "Final value"<< sum_r[0]
                  <<std::endl;
    }

    if (print)
    {
        std::cout
            << std::left << std::setw(24) << "G barrier BA"
            << std::left << std::setw(24) << *minmax.first*1E-9
            << std::left << std::setw(24) << *minmax.second*1E-9
            << std::left << std::setw(24) << average*1E-9
            << std::endl
            << std::fixed;
    }
    

    free(sum);
    
    
}

void local_barrier_test_buff_acc(sycl::queue &Q, int size, int block_size, bool print, int iter)
{

    timer time;

    size = size*size;

    TYPE * sum = (TYPE *)malloc(size*sizeof(TYPE)); 

    std::fill(sum,sum+(size),0);
    
    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};
    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the matrix size change block size to matrix size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    sycl::buffer<TYPE , 1> sum_buff(sum,size);

    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();

        Q.submit([&](sycl::handler& cgh){
            auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

            cgh.parallel_for<class local_barrier_ba>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

                auto k = it.get_global_id(0);

                for (size_t l = 0; l < 1024; l++)
                {
                    sum_acc[k]+= 1;
                }

                it.barrier(sycl::access::fence_space::local_space);
            
            });

        });
        
        Q.wait();

        time.end_timer();

        timings[i] = time.duration();
    }
    
    auto minmax = std::minmax_element(timings, timings+iter);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    auto sum_r = sum_buff.get_host_access();

    if (sum_r[0]!= 1024*iter)
    {
        std::cout << "Verification failed "
                << "Expected value "<< 1024*iter
                << "Final value"<< sum_r[0]
                << std::endl;
    }

    if (print)
    {
        std::cout
            << std::left << std::setw(24) << "L barrier BA"
            << std::left << std::setw(24) << *minmax.first*1E-9
            << std::left << std::setw(24) << *minmax.second*1E-9
            << std::left << std::setw(24) << average*1E-9
            << std::endl
            << std::fixed;
    }
    
    free(sum);

    
}