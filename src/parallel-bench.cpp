#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>
#include <algorithm>


#ifndef TYPE
#define TYPE double
#endif

#include "timer.hpp"
#include "parallel-bench.hpp"

//#define REDUCTION_IN_SYCL

#if defined DPCPP 
    extern SYCL_EXTERNAL int rand(void);
#endif

using namespace cl;

double delay_time()
{
    timer time;
    time.start_timer();
    TYPE sum = 0;
    for (size_t l = 0; l < 1024; l++)
    {
        sum += 1;
        
        if (sum < 0)
        {
            sum = 0;
        }
        
    }
    time.end_timer();
    auto kernel_offload_time = time.duration()/(1E+9);

    return kernel_offload_time;

}

// memory allocations

void memory_alloc(sycl::queue &Q, int size, int block_size , bool print, int iter)
{
    timer time;

    int i;

    auto N = static_cast<size_t>(size);

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    for (i = 0; i < iter; i++)
    {
        auto m = (TYPE*)std::malloc(size*size*sizeof(TYPE)); 

        auto a = (TYPE*)std::malloc(size*size*sizeof(TYPE));

        std::fill(m,m+(size*size),0.0);
        std::fill(a,a+(size*size),1.0);

        sycl::range<1> global{N*N};

        //sycl::buffer<TYPE , 1> m_buff((TYPE*)m,size*size);
        //sycl::buffer<TYPE , 1> a_buff((TYPE*)a,size*size);

        sycl::buffer<TYPE , 1> m_buff(global);
        sycl::buffer<TYPE , 1> a_buff(global);

        Q.submit([&](sycl::handler& cgh){

            sycl::accessor m_acc(m_buff, cgh, sycl::write_only, sycl::no_init);
            sycl::accessor a_acc(a_buff, cgh, sycl::write_only, sycl::no_init);

            //auto m_acc = m_buff.get_access<sycl::access::mode::discard_write>(cgh);
            //auto a_acc = a_buff.get_access<sycl::access::mode::discard_write>(cgh);

            cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

                const int k = it.get_id(0);

                m_acc[k] = 0.0 ;
                a_acc[k] = 1.0;
            
            });

        });
        Q.wait();

        time.start_timer();

        Q.submit([&](sycl::handler& cgh){

            auto m_acc = m_buff.get_access<sycl::access::mode::write>(cgh);
            auto a_acc = a_buff.get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

                const int k = it.get_id(0);

                m_acc[k] = a_acc[k];
            
            });

        });

        Q.wait();
        
        time.end_timer();
        timings[i] = time.duration();
        free(m);
        free(a);
    }
    
    auto minmax = std::minmax_element(timings, timings+iter);

    double bandwidth = 1.0E-6 * 2 *size*size*sizeof(TYPE) / (*minmax.first*1E-9);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
          << std::left << std::setw(24) << "memory B&A"
          << std::left << std::setw(24) << bandwidth
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl;
    }

    free(timings);


}


void host_memory_alloc(sycl::queue &Q, int size, int block_size , bool print, int iter)
{

    timer time;

    timer time1;
    
    int i;

    auto N = static_cast<size_t>(size);

    auto timings_alloc = (double*)std::malloc(sizeof(double)*iter);

    for ( i = 0; i < 10; i++)
    {
        time.start_timer();

        auto m_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();

        free(m_host,Q);

        time.end_timer();

        timings_alloc[i] = time.duration();

    }
/*
    auto minmax = std::minmax_element(timings_alloc, timings_alloc+iter);

    double average = std::accumulate(timings_alloc, timings_alloc+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
          << std::left << std::setw(24) << "Host memory alloc"
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl;
    }
*/    
    
    auto timings = (double*)std::malloc(sizeof(double)*iter);

    auto timings_nd = (double*)std::malloc(sizeof(double)*iter);

    for (size_t i = 0; i < iter; i++)
    {
        auto m_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();

        auto a_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();
        std::fill(a_host,a_host+(size*size),1);

        sycl::range<1> global{N*N};
        time1.start_timer();

        Q.parallel_for<class host_memory>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int k = it.get_id(0);

            m_host[k] = a_host[k];

        
        });
        Q.wait();

        time1.end_timer();
        timings[i] = time1.duration();
        free(m_host,Q);
        free(a_host,Q);
    }

    auto minmax = std::minmax_element(timings, timings+iter);

    double bandwidth = 1.0E-6 * 2 *size*size*sizeof(TYPE) / (*minmax.first*1E-9);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
          << std::left << std::setw(24) << "Host memory (range)"
          << std::left << std::setw(24) << bandwidth
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl;
    }
    
    ///////////////////////////////////////////

    for (size_t i = 0; i < iter; i++)
    {
        auto m_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();

        auto a_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();
        std::fill(a_host,a_host+(size*size),1);

        sycl::range<1> global{N*N};
        auto N_b = static_cast<size_t>(block_size);
        if (block_size > size)
        {
            std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
            N_b = N;
        }
        sycl::range<1> local{N_b};

        time1.start_timer();

        Q.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            const int k = it.get_global_id(0);

            m_host[k] = a_host[k];

        
        });
        Q.wait();

        time1.end_timer();
        timings_nd[i] = time1.duration();
        free(m_host,Q);
        free(a_host,Q);
    }

    minmax = std::minmax_element(timings_nd, timings_nd+iter);

    bandwidth = 1.0E-6 * 2 *size*size*sizeof(TYPE) / (*minmax.first*1E-9);

    average = std::accumulate(timings_nd, timings_nd+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
          << std::left << std::setw(24) << "Host memory (ndrange)"
          << std::left << std::setw(24) << bandwidth
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl;
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

    for ( i = 0; i < 10; i++)
    {
        time.start_timer();
        auto m_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();
        free(m_shared,Q);
        time.end_timer();

        timings_alloc[i] = time.duration();
    }
/*
    auto minmax = std::minmax_element(timings_alloc, timings_alloc+iter);

    double average = std::accumulate(timings_alloc, timings_alloc+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
          << std::left << std::setw(24) << "shared memory alloc"
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl;
    }
    
*/
    auto timings = (double*)std::malloc(sizeof(double)*iter);

    auto timings_nd = (double*)std::malloc(sizeof(double)*iter);

    for (size_t i = 0; i < iter; i++)
    {
        sycl::range<1> global{N*N};
        auto m_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();

        auto a_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();
        std::fill(a_shared,a_shared+(size*size),1);

        time1.start_timer();
        
        Q.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int k = it.get_id(0);

            m_shared[k] = a_shared[k];

        
        });
        Q.wait();

        time1.end_timer();
        timings[i] = time1.duration();
        free(m_shared,Q);
        free(a_shared,Q);
    }

    auto minmax = std::minmax_element(timings, timings+iter);

    double bandwidth = 1.0E-6 * 2 *size*size*sizeof(TYPE) / (*minmax.first*1E-9);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
          << std::left << std::setw(24) << "shared memory (range)"
          << std::left << std::setw(24) << bandwidth
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl;
    }

    ///////////////////////////////////////////////
    
    for (size_t i = 0; i < iter; i++)
    {
        sycl::range<1> global{N*N};
        auto N_b = static_cast<size_t>(block_size);
        if (block_size > size)
        {
            std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
            N_b = N;
        }
        sycl::range<1> local{N_b};

        auto m_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();

        auto a_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();
        std::fill(a_shared,a_shared+(size*size),1);

        time1.start_timer();
        
        Q.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            const int k = it.get_global_id(0);

            m_shared[k] = a_shared[k];

        
        });
        Q.wait();

        time1.end_timer();
        timings_nd[i] = time1.duration();
        free(m_shared,Q);
        free(a_shared,Q);
    }

    minmax = std::minmax_element(timings_nd, timings_nd+iter);

    bandwidth = 1.0E-6 * 2 *size*size*sizeof(TYPE) / (*minmax.first*1E-9);

    average = std::accumulate(timings_nd, timings_nd+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
          << std::left << std::setw(24) << "shared memory (ndrange)"
          << std::left << std::setw(24) << bandwidth
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl;
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

    for ( i = 0; i < 10; i++)
    {
        time.start_timer();
        auto m_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();
        free(m_device,Q);
        time.end_timer();

        timings_alloc[i] = time.duration();

    }
/*
    auto minmax = std::minmax_element(timings_alloc, timings_alloc+iter);

    double average = std::accumulate(timings_alloc, timings_alloc+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
          << std::left << std::setw(24) << "Device memory alloc"
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl;
    }
*/
    auto timings = (double*)std::malloc(sizeof(double)*iter);

    auto timings_nd = (double*)std::malloc(sizeof(double)*iter);

    for (size_t i = 0; i < iter; i++)
    {
        sycl::range<1> global{N*N};
        auto m_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();

        auto a_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();
        std::fill(a_device,a_device+(size*size),1);

        time1.start_timer();

        Q.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int k = it.get_id(0);

            m_device[k] = a_device[k];

        
        });
        Q.wait();

        time1.end_timer();
        timings[i] = time1.duration();
        free(m_device,Q);
        free(a_device,Q);
    }

    auto minmax = std::minmax_element(timings, timings+iter);

    double bandwidth = 1.0E-6 * 2 * size*size*sizeof(TYPE) / (*minmax.first*1E-9);

    double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
          << std::left << std::setw(24) << "Device memory (range)"
          << std::left << std::setw(24) << bandwidth
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl;
    }

    /////////////////////////////////////////////////

    for (size_t i = 0; i < iter; i++)
    {
        sycl::range<1> global{N*N};
        auto N_b = static_cast<size_t>(block_size);
        if (block_size > size)
        {
            std::cout << "Given input block size is greater than the global size changing block size to global size \n" << std::endl;
            N_b = N;
        }
        sycl::range<1> local{N_b};

        auto m_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();

        auto a_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();
        std::fill(a_device,a_device+(size*size),1);

        time1.start_timer();

        Q.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            const int k = it.get_global_id(0);

            m_device[k] = a_device[k];

        
        });
        Q.wait();

        time1.end_timer();
        timings_nd[i] = time1.duration();
        free(m_device,Q);
        free(a_device,Q);
    }

    minmax = std::minmax_element(timings_nd, timings_nd+iter);

    bandwidth = 1.0E-6 * 2 *size*size*sizeof(TYPE) / (*minmax.first*1E-9);

    average = std::accumulate(timings_nd, timings_nd+iter, 0.0) / (double)(iter);

    if (print)
    {
        std::cout
          << std::left << std::setw(24) << "Device memory (ndrange)"
          << std::left << std::setw(24) << bandwidth
          << std::left << std::setw(24) << *minmax.first*1E-9
          << std::left << std::setw(24) << *minmax.second*1E-9
          << std::left << std::setw(24) << average*1E-9
          << std::endl;
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

    sycl::range<1> global{N};    

    auto timings = (double*)std::malloc(sizeof(double)*iter);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        const int DIM =1;

        sycl::buffer<TYPE , 1> sum_buff((TYPE*)sum,size*size);

        int i;

        

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();

            Q.submit([&](sycl::handler& cgh){
                auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for<>(sycl::range<DIM>(global), [=](sycl::item<DIM>it){
                    
                    auto k = it.get_id(0);

                    for (size_t l = 0; l < 1024; l++)
                    {
                        sum_acc[k] += 1;
                        
                    }
                    
                
                });

            });
            Q.wait();

            time.end_timer();

            timings[i] = time.duration();
        }

        auto minmax = std::minmax_element(timings, timings+iter);

        double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

        auto sum_r = sum_buff.get_access<sycl::access::mode::read>();

        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<std::endl;
        }

        if (print)
        {
            std::cout
                << std::left << std::setw(24) << "range_BA"
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
        const int DIM =2;

        sycl::buffer<TYPE , 1> sum_buff((TYPE*)sum,size*size);

        int i;

        for ( i = 0; i < iter; i++)
        {
            time.start_timer();

            Q.submit([&](sycl::handler& cgh){
                auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for<>(sycl::range<DIM>(global), [=](sycl::item<DIM>it){

                    auto k = it.get_id(0);
                    auto k1 = it.get_id(1);

                    for (size_t l = 0; l < 1024; l++)
                    {
                        sum_acc[k*N+k1] += 1;
                        
                    }
                    
                
                });

            });
            Q.wait();

            time.end_timer();

            timings[i] = time.duration();
        }
        
        auto minmax = std::minmax_element(timings, timings+iter);

        double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

        auto sum_r = sum_buff.get_access<sycl::access::mode::read>();


        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<std::endl;
        }


        if (print)
        {
            std::cout
                << std::left << std::setw(24) << "range_BA"
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

    free((TYPE*)sum);
         
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
        const int DIM =1;
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

            Q.submit([&](sycl::handler& cgh){
                auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for<>(sycl::nd_range<DIM>(global,local), [=](sycl::nd_item<DIM>it){
                    
                    auto k = it.get_global_id(0);

                    for (size_t l = 0; l < 1024; l++)
                    {
                        sum_acc[k] += 1;
                        
                    }
                
                });

            });
            Q.wait();

            time.end_timer();

            timings[i] = time.duration();
        }

        auto minmax = std::minmax_element(timings, timings+iter);

        double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

        auto sum_r = sum_buff.get_access<sycl::access::mode::read>();


        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<  std::endl;
        }

        if (print)
        {
            std::cout
                << std::left << std::setw(24) << "ndrange_BA"
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
        const int DIM =2;
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

            Q.submit([&](sycl::handler& cgh){
                auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for<>(sycl::nd_range<DIM>(global,local), [=](sycl::nd_item<DIM>it){

                    auto k = it.get_global_id(0);
                    auto k1 = it.get_global_id(1);

                    for (size_t l = 0; l < 1024; l++)
                    {
                        sum_acc[k*N+k1] += 1;
                        
                    }
                    
                
                });

            });
            Q.wait();

            time.end_timer();

            timings[i] = time.duration();
        }

        auto minmax = std::minmax_element(timings, timings+iter);

        double average = std::accumulate(timings, timings+iter, 0.0) / (double)(iter);

        auto sum_r = sum_buff.get_access<sycl::access::mode::read>();


        if (sum_r[1]!= 1024*iter)
        {
            std::cout << "Verification failed "
                      << "Expected value "<< 1024*iter
                      << "Final value"<< sum_r[1]
                      <<std::endl;
        }

        if (print)
        {
            std::cout
                << std::left << std::setw(24) << "ndrange_BA"
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
    
    auto sum_r = sum_buff.get_access<sycl::access::mode::read>();


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

    auto sum_r = sum_buff.get_access<sycl::access::mode::read>();

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

    auto sum_r = sum_buff.get_access<sycl::access::mode::read>();

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