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

void host_memory_alloc(sycl::queue &Q, int size, int block_size , bool print, int iter)
{

    timer time;

    timer time1;
    
    int i;

    auto N = static_cast<size_t>(size);

    auto timings_alloc = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();

        auto m_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();

        free(m_host,Q);

        time.end_timer();

        timings_alloc[i] = time.duration();

    }

    if (print)
    {
        print_results(timings_alloc, iter, size, "Host memory alloc",1, 1);
    }
   
    auto timings = (double*)std::malloc(sizeof(double)*iter);

    auto timings_nd = (double*)std::malloc(sizeof(double)*iter);
  
    auto m_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();
    auto a_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();

    sycl::range<1> global{N*N};

    init_arrays(Q, m_host, a_host, global);

    for (size_t i = 0; i < iter; i++)
    {   

        time1.start_timer();
        kernel_copy(Q, m_host, a_host, global);
        time1.end_timer();

        timings[i] = time1.duration();
        
    }

    if (print)
    {
        print_results(timings, iter, size, "Host memory (r)",1, 1);
    }
    
    ///////////////////////////////////////////

    Q.wait();

    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    init_arrays(Q, m_host, a_host, global);

    for (size_t i = 0; i < iter; i++)
    {

        time1.start_timer();
        kernel_copy(Q,m_host,a_host,global,local);
        time1.end_timer();
        timings_nd[i] = time1.duration();
        
    }
    
    sycl::free(m_host,Q);
    sycl::free(a_host,Q);

    if (print)
    {
        print_results(timings_nd, iter, size, "Host memory (ndr)",1, 1);
    }

    free(timings);
    free(timings_alloc);
    free(timings_nd);

}

void shared_memory_alloc(sycl::queue &Q, int size, int block_size ,bool print, int iter)
{

    timer time;

    timer time1;

    int i;

    auto N = static_cast<size_t>(size);

    auto timings_alloc = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();
        auto m_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();
        free(m_shared,Q);
        time.end_timer();

        timings_alloc[i] = time.duration();
    }

    if (print)
    {
        print_results(timings_alloc, iter, size, "Shared memory alloc",1, 1);
    }
    

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    auto timings_nd = (double*)std::malloc(sizeof(double)*iter);

    sycl::range<1> global{N*N};
    auto m_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();
    auto a_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();
    
    init_arrays(Q, m_shared, a_shared, global);

    for (size_t i = 0; i < iter; i++)
    {
        Q.wait();

        time1.start_timer();
        kernel_copy(Q, m_shared, a_shared, global);
        time1.end_timer();

        timings[i] = time1.duration();
        
    }

    if (print)
    {
        print_results(timings, iter, size, "Shared memory (r)",1, 1);
    }

    ///////////////////////////////////////////////

    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    init_arrays(Q, m_shared, a_shared, global);
    
    for (size_t i = 0; i < iter; i++)
    {

        time1.start_timer();
        kernel_copy(Q,m_shared,a_shared,global,local);
        time1.end_timer();

        timings_nd[i] = time1.duration();
        
    }

    sycl::free(m_shared,Q);
    sycl::free(a_shared,Q);

    if (print)
    {
        print_results(timings_nd, iter, size, "Shared memory (ndr)",1, 1);
    }

    free(timings);
    free(timings_alloc);
    free(timings_nd);

}

void device_memory_alloc(sycl::queue &Q, int size, int block_size ,bool print, int iter)
{

    timer time;
    timer time1;

    int i;
    
    auto N = static_cast<size_t>(size);

    auto timings_alloc = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();
        auto m_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();
        free(m_device,Q);
        time.end_timer();

        timings_alloc[i] = time.duration();

    }

    if (print)
    {
        print_results(timings_alloc, iter, size, "Device memory alloc",1, 1);
    }

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    auto timings_nd = (double*)std::malloc(sizeof(double)*iter);

    sycl::range<1> global{N*N};
    auto m_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();
    auto a_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();

    init_arrays(Q, m_device, a_device, global);
    
    Q.wait();

    for (size_t i = 0; i < iter; i++)
    {

        time1.start_timer();
        kernel_copy(Q, m_device, a_device, global);
        time1.end_timer();

        timings[i] = time1.duration();
        
    }

    if (print)
    {
        print_results(timings, iter, size, "Device memory (r)",1, 1);
    }

    /////////////////////////////////////////////////

    auto N_b = static_cast<size_t>(block_size);
    if (block_size > size)
    {
        std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
        N_b = N;
    }
    sycl::range<1> local{N_b};

    init_arrays(Q, m_device, a_device, global);

    for (size_t i = 0; i < iter; i++)
    {

        time1.start_timer();
        kernel_copy(Q,m_device,a_device,global,local);
        time1.end_timer();

        timings_nd[i] = time1.duration();    
    }

    sycl::free(m_device,Q);
    sycl::free(a_device,Q);

    if (print)
    {
        print_results(timings_nd, iter, size, "Device memory (ndr)",1, 1);
    }

    free(timings);
    free(timings_alloc);
    free(timings_nd);

}


// sycl::range constuct


void range_with_usm(sycl::queue &Q, int size, int dim, bool print, int iter)
{

    /*
    * creates a SYCL parallel region using <range> contruct for a given problem size 
    * the dimensions of the range contruct can also be specified, the parameter <dim> 
    * takes values 1 or 2. 
    * 
    * This benchmark tests the overhead incurred for the thread creation. each thread 
    * computes a small kernel, which corresponds to dealy time. This benchmark uses USM
    * to store the variables.
    * 
    */

    timer time;

    TYPE * sum = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(sum,sum+(size*size),0);

    auto N = static_cast<size_t>(size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        const int DIM = 1;

        int i;

        

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();

            Q.parallel_for<>(sycl::range<DIM>(global), [=](sycl::item<DIM>it){

                auto k = it.get_id(0);

                for (size_t l = 0; l < 1024; l++)
                {
                    sum[k] += 1;

                }

            
            });
            Q.wait();

            time.end_timer();

            timings[i] = time.duration();
            
        }
        

        auto minmax = std::minmax_element(timings, timings+iter);

        double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);
        
        if (sum[1] != 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum[1]
                      <<std::endl;
        }

        if (print)
        {
            std::cout
                << std::left << std::setw(24) << "range_USM"
                << std::left << std::setw(24) << 1
                << std::left << std::setw(24) << *minmax.first*1E-9
                << std::left << std::setw(24) << *minmax.second*1E-9
                << std::left << std::setw(24) << average*1E-9
                << std::endl
                << std::fixed;
        }
        
        


    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        const int DIM = 2;
        int i;

        

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();

            Q.parallel_for<>(sycl::range<DIM>(global), [=](sycl::item<DIM>it){
                
                auto k = it.get_id(0);
                auto k1 = it.get_id(1);

                for (size_t l = 0; l < 1024; l++)
                {
                    sum[k*N+k1] += 1;         
                    
                }
                
            
            });
            Q.wait();

            time.end_timer();

            timings[i] = time.duration();
        }

        
        auto minmax = std::minmax_element(timings, timings+iter);

        double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);


        if (sum[1] != 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum[1]
                      <<std::endl;
        }

        if (print)
        {
            std::cout
                << std::left << std::setw(24) << "range_USM"
                << std::left << std::setw(24) << 2
                << std::left << std::setw(24) << *minmax.first*1E-9
                << std::left << std::setw(24) << *minmax.second*1E-9
                << std::left << std::setw(24) << average*1E-9
                << std::endl
                << std::fixed;
        }
        
        
    

    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }
    
    free((TYPE*)sum,Q);
    
}

// sycl::nd_range constuct


void nd_range_with_usm(sycl::queue &Q, int size, int block_size ,int dim, bool print, int iter)
{

    /*
    * creates a SYCL parallel region using <nd_range> contruct for a given problem size 
    * the dimensions of the range contruct can also be specified, the parameter <dim> 
    * takes values 1 or 2. 
    * 
    * This benchmark tests the overhead incurred for the thread creation. each thread 
    * computes a small kernel, which corresponds to delay time. This benchmark uses USM
    * to store the variables.
    * 
    */

    timer time;

    TYPE * sum = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(sum,sum+(size*size),0);

    auto N = static_cast<size_t>(size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        const int DIM = 1;
        int i;

        auto N_b = static_cast<size_t>(block_size);
        if (block_size > size)
        {
            std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
            N_b = N;
        }
        sycl::range<1> local{N_b};

        

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();

            Q.parallel_for<>(sycl::nd_range<DIM>(global,local), [=](sycl::nd_item<DIM>it){

                auto k = it.get_global_id(0);

                for (size_t l = 0; l < 1024; l++)
                {
                    sum[k] += 1;
                    
                }
                
            
            });
            Q.wait();

            time.end_timer();

            timings[i] = time.duration();
        }

        auto minmax = std::minmax_element(timings, timings+iter);

        double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);


        if (sum[1] != 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value "<< sum[1]
                      <<std::endl;
        }


        if (print)
        {
            std::cout
                << std::left << std::setw(24) << "ndrange_USM"
                << std::left << std::setw(24) << 1
                << std::left << std::setw(24) << *minmax.first*1E-9
                << std::left << std::setw(24) << *minmax.second*1E-9
                << std::left << std::setw(24) << average*1E-9
                << std::endl
                << std::fixed;
        }
        
        


    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        const int DIM = 2;
        int i;

        auto N_b = static_cast<size_t>(block_size);
        if (block_size > size)
        {
            std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
            N_b = N;
        }
        sycl::range<2> local{N_b,N_b};

        

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();

            Q.parallel_for<>(sycl::nd_range<DIM>(global,local), [=](sycl::nd_item<DIM>it){

                auto k = it.get_global_id(0);
                auto k1 = it.get_global_id(1);

                for (size_t l = 0; l < 1024; l++)
                {
                    sum[k*N+k1] += 1;
                    
                }
                
            
            });
            Q.wait();

            time.end_timer();

            timings[i] = time.duration();
        }

        auto minmax = std::minmax_element(timings, timings+iter);

        double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

        if (sum[1] != 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum[1]
                      << std::endl;
        }

        if (print)
        {
            std::cout
                << std::left << std::setw(24) << "ndrange_USM"
                << std::left << std::setw(24) << 2
                << std::left << std::setw(24) << *minmax.first*1E-9
                << std::left << std::setw(24) << *minmax.second*1E-9
                << std::left << std::setw(24) << average*1E-9
                << std::endl
                << std::fixed;
        }

    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }
    
    free(sum,Q);    
    
} 


// reduction 

void reduction_with_atomics_usm(sycl::queue &Q, int size, bool print, int iter)
{
    timer time;

    size = size*size;

    auto m_shared = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    std::fill(m_shared,m_shared+size,1.0);
    auto sum = sycl::malloc_shared<TYPE>(1*sizeof(TYPE),Q); Q.wait();

    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};

    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();

        Q.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            auto j = it.get_id(0);

            auto v = sycl::atomic_ref<TYPE, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(
            sum[0]);

            
            v.fetch_add(m_shared[j]);
            
        });

        Q.wait();

        time.end_timer();

        timings[i] = time.duration();
    }   

    auto minmax = std::minmax_element(timings, timings+iter);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    if (sum[0]!= size*iter)
    {
        std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum[0]
                      <<std::endl;
    }

    if (print)
    {
        std::cout
            << std::left << std::setw(24) << "atomics_USM"
            << std::left << std::setw(24) << *minmax.first*1E-9
            << std::left << std::setw(24) << *minmax.second*1E-9
            << std::left << std::setw(24) << average*1E-9
            << std::endl
            << std::fixed;
    }
    
    
    
    
    free(m_shared,Q);
}


////////////////////////////// change it to usm 
#if defined(REDUCTION_IN_SYCL) 

void reduction_with_usm(sycl::queue &Q, int size, int block_size, bool print, int iter)
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

            auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

#if defined(DPCPP) 
            auto sum_red = sycl::reduction(sum_buff, cgh,sycl::plus<TYPE>());
#else
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


void global_barrier_test_usm(sycl::queue &Q, int size, int block_size, bool print, int iter)
{
    
    timer time;

    size = size*size;

    TYPE * sum = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();

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

    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {
        time.start_timer();

        Q.parallel_for<class global_barrier_usm>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            auto k = it.get_global_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum[k]+= 1;
            }

            it.barrier();
        
        });
        Q.wait();

        time.end_timer();

        timings[i] = time.duration();
    }

    auto minmax = std::minmax_element(timings, timings+iter);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    if (sum[0]!= 1024*iter)
    {
        std::cout << "Verification failed "
                  << "Expected value "<< 1024*iter
                  << "Final value"<< sum[0]
                  <<std::endl;
    }

    if (print)
    {
        std::cout
            << std::left << std::setw(24) << "G barrier USM"
            << std::left << std::setw(24) << *minmax.first*1E-9
            << std::left << std::setw(24) << *minmax.second*1E-9
            << std::left << std::setw(24) << average*1E-9
            << std::endl
            << std::fixed;
    }
    
    

    free(sum,Q);
    
    
}


void local_barrier_test_usm(sycl::queue &Q, int size, int block_size, bool print, int iter)
{

    timer time;

    size = size*size;

    TYPE * sum = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();

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

    int i;

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < iter; i++)
    {

        time.start_timer();

        Q.parallel_for<class local_barrier_usm>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            auto k = it.get_global_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum[k]+= 1;
            }


            it.barrier(sycl::access::fence_space::local_space);
        
        });
        Q.wait();

        time.end_timer();

        timings[i] = time.duration();
    }
    
    auto minmax = std::minmax_element(timings, timings+iter);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    if (sum[0]!= 1024*iter)
    {
        std::cout << "Verification failed "
                  << "Expected value "<< 1024*iter
                  << "Final value"<< sum[0]
                  <<  std::endl;
    }

    if (print)
    {
        std::cout
            << std::left << std::setw(24) << "L barrier USM"
            << std::left << std::setw(24) << *minmax.first*1E-9
            << std::left << std::setw(24) << *minmax.second*1E-9
            << std::left << std::setw(24) << average*1E-9
            << std::endl
            << std::fixed;
    }
    

    free(sum,Q);

}
