#include <iostream>
#include <math.h>
#include <vector>
#include <CL/sycl.hpp>
#include <getopt.h>
#include <assert.h>
#include <sys/time.h>
#include <chrono>


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

void host_memory_alloc(sycl::queue &Q, int size)
{

    timer time;

    timer time1;
    
    float time_fill = 0;
    int i;

    auto N = static_cast<size_t>(size);

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {
        auto m_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();
        free(m_host,Q);
    }
    
    time.end_timer();

    auto host_alloc_time = time.duration();

    for (size_t i = 0; i < 10; i++)
    {
        auto m_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();

        auto a_host = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();
        std::fill(a_host,a_host+(size*size),1);

        sycl::range<1> global{N*N};
        time1.start_timer();

        Q.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            auto k = it.get_id(0);

            m_host[k] = a_host[k];

        
        });
        Q.wait();

        time1.end_timer();
        time_fill += time1.duration();
        free(m_host,Q);
        free(a_host,Q);
    }

    std::cout << "Total time taken for the host memory allocation with sycl "<< host_alloc_time/10 << " nanoseconds\n" 
                 "fill time is "<< time_fill/(10*1E3)<< " microseconds"<<std::endl;

}

void shared_memory_alloc(sycl::queue &Q, int size)
{

    timer time;

    timer time1;
    
    float time_fill = 0;

    int i;

    auto N = static_cast<size_t>(size);

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {
        auto m_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();
        free(m_shared,Q);
    }
    
    time.end_timer();

    auto shared_alloc_time = time.duration();

    for (size_t i = 0; i < 10; i++)
    {
        sycl::range<1> global{N*N};
        auto m_shared = sycl::malloc_shared<TYPE>(size*size,Q); Q.wait();

        auto a_shared = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();
        std::fill(a_shared,a_shared+(size*size),1);

        time1.start_timer();
        
        Q.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            auto k = it.get_id(0);

            m_shared[k] = a_shared[k];

        
        });
        Q.wait();

        time1.end_timer();
        time_fill += time1.duration();
        free(m_shared,Q);
        free(a_shared,Q);
    }

    std::cout << "Total time taken for the shared memory allocation with sycl "<< shared_alloc_time/10 << " nanoseconds\n" 
                "fill time is "<< time_fill/(10*1E3)<< " microseconds"<<std::endl;

}

void device_memory_alloc(sycl::queue &Q, int size)
{

    timer time;
    timer time1;

    int i;
    
    float time_fill = 0;
    auto N = static_cast<size_t>(size);

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {
        auto m_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();
        free(m_device,Q);
    }
    
    time.end_timer();

    auto device_alloc_time = time.duration();
    for (size_t i = 0; i < 10; i++)
    {
        sycl::range<1> global{N*N};
        auto m_device = sycl::malloc_device<TYPE>(size*size,Q); Q.wait();

        auto a_device = sycl::malloc_host<TYPE>(size*size,Q); Q.wait();
        std::fill(a_device,a_device+(size*size),1);

        time1.start_timer();

        Q.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            auto k = it.get_id(0);

            m_device[k] = a_device[k];

        
        });
        Q.wait();

        time1.end_timer();
        time_fill += time1.duration();
        free(m_device,Q);
        free(a_device,Q);
    }

    std::cout << "Total time taken for the device memory allocation with sycl "<< device_alloc_time/10 << " nanoseconds\n" 
                 "fill time is "<< time_fill/(10*1E3)<< " microseconds"<<std::endl;

}


// sycl::range constuct


void range_with_usm(sycl::queue &Q, int size, int dim)
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

    volatile TYPE * sum = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(sum,sum+(size*size),0);

    auto N = static_cast<size_t>(size);

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        const int DIM = 1;

        int i;

        time.start_timer();

        for ( i = 0; i < 10; i++)
        {
            Q.parallel_for<>(sycl::range<DIM>(global), [=](sycl::item<DIM>it){

                auto k = it.get_id(0);

                for (size_t l = 0; l < 1024; l++)
                {
                    sum[k] += 1;

                }

            
            });
            Q.wait();

            
        }
        

        time.end_timer();

       
        auto kernel_offload_time = time.duration();
        
        if (sum[1] != 1024*10)
        {
            std::cout << "Verification failed"<< std::endl;
        }
        std::cout << "Total time taken for the execution of range parallel construct with "<< dim <<" dim \n and USM is "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;


    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        const int DIM = 2;
        int i;

        time.start_timer();

        for ( i = 0; i < 10; i++)
        {
            Q.parallel_for<>(sycl::range<DIM>(global), [=](sycl::item<DIM>it){
                
                auto k = it.get_id(0);
                auto k1 = it.get_id(1);

                for (size_t l = 0; l < 1024; l++)
                {
                    sum[k*N+k1] += 1;         
                    
                }
                
            
            });
            Q.wait();
        }
        

        time.end_timer();


        if (sum[1] != 1024*10)
        {
            std::cout << "Verification failed"<< std::endl;
        }

        auto kernel_offload_time = time.duration();
        std::cout << "Total time taken for the execution of range parallel construct with "<< dim <<" dim \n and USM is "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    

    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }
    
    free((TYPE*)sum,Q);
    
}

void range_with_buff_acc(sycl::queue &Q, int size, int dim)
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

    volatile TYPE * sum = (TYPE *)malloc(size*size*sizeof(TYPE)); 

    std::fill(sum,sum+(size*size),0);

    const int DIM = dim;
    
    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};    

    if (dim == 1)
    {
        sycl::range<1> global{N*N};
        const int DIM =1;

        sycl::buffer<TYPE , 1> sum_buff((TYPE*)sum,size*size);

        int i;

        time.start_timer();

        for ( i = 0; i < 10; i++)
        {
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
        }
        

        time.end_timer();


        auto sum_r = sum_buff.get_access<sycl::access::mode::read>();

        if (sum_r[1]!= 1024*10)
        {
            std::cout << "Verification failed"<< std::endl;
        }

        auto kernel_offload_time = time.duration();
        std::cout << "Total time taken for the execution of range parallel construct with " << dim << " dim \n and buff and acc is " << kernel_offload_time / (10 * 1E9) << " seconds\n"<< std::endl;
    }
    else if (dim == 2)
    {
        sycl::range<2> global{N,N};
        const int DIM =2;

        sycl::buffer<TYPE , 1> sum_buff((TYPE*)sum,size*size);

        int i;

        time.start_timer();

        for ( i = 0; i < 10; i++)
        {
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
        }
        

        time.end_timer();

        auto sum_r = sum_buff.get_access<sycl::access::mode::read>();


        if (sum_r[1]!= 1024*10)
        {
            std::cout << "Verification failed"<< std::endl;
        }

        auto kernel_offload_time = time.duration();
        std::cout << "Total time taken for the execution of range parallel construct with "<< dim <<" dim \n and buff and acc is "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    
    } 
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }

    free((TYPE*)sum);
         
}


// sycl::nd_range constuct


void nd_range_with_usm(sycl::queue &Q, int size, int block_size ,int dim)
{

    /*
    * creates a SYCL parallel region using <nd_range> contruct for a given problem size 
    * the dimensions of the range contruct can also be specified, the parameter <dim> 
    * takes values 1 or 2. 
    * 
    * This benchmark tests the overhead incurred for the thread creation. each thread 
    * computes a small kernel, which corresponds to dealy time. This benchmark uses USM
    * to store the variables.
    * 
    */

    std::cout<< "\n Local range of the <nd_range> construct is: "<< block_size << std::endl;

    timer time;

    TYPE * sum = sycl::malloc_shared<TYPE>(size*size*sizeof(TYPE),Q); Q.wait();

    std::fill(sum,sum+(size*size),0);

    auto N = static_cast<size_t>(size);

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

        time.start_timer();

        for ( i = 0; i < 10; i++)
        {
            Q.parallel_for<>(sycl::nd_range<DIM>(global,local), [=](sycl::nd_item<DIM>it){

                auto k = it.get_global_id(0);

                for (size_t l = 0; l < 1024; l++)
                {
                    sum[k] += 1;
                    
                }
                
            
            });
            Q.wait();
        }
        

        time.end_timer();


        if (sum[1] != 1024*10)
        {
            std::cout << "Verification failed"<< std::endl;
        }

        auto kernel_offload_time = time.duration();
        std::cout << "Total time taken for the execution of nd_range parallel construct with "<< dim <<" dim \n and USM is "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;


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

        time.start_timer();

        for ( i = 0; i < 10; i++)
        {
            Q.parallel_for<>(sycl::nd_range<DIM>(global,local), [=](sycl::nd_item<DIM>it){

                auto k = it.get_global_id(0);
                auto k1 = it.get_global_id(1);

                for (size_t l = 0; l < 1024; l++)
                {
                    sum[k*N+k1] += 1;
                    
                }
                
            
            });
            Q.wait();
        }
        

        time.end_timer();


        if (sum[1] != 1024*10)
        {
            std::cout << "Verification failed"<< std::endl;
        }

        auto kernel_offload_time = time.duration();
        std::cout << "Total time taken for the execution of nd_range parallel construct with "<< dim <<" dim \n and USM is "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    

    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }
    
    free(sum,Q);    
    
} 

void nd_range_with_buff_acc(sycl::queue &Q, int size, int block_size ,int dim)
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

    std::cout<< "\n Local range of the <nd_range> construct is: "<< block_size << std::endl;

    timer time;

    TYPE * sum = (TYPE *)malloc(size*size*sizeof(TYPE)); 

    std::fill(sum,sum+(size*size),0);

    const int DIM = dim;
    
    auto N = static_cast<size_t>(size);

    sycl::range<1> global{N};    

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


        time.start_timer();

        for ( i = 0; i < 10; i++)
        {
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
        }
        

        time.end_timer();


        auto sum_r = sum_buff.get_access<sycl::access::mode::read>();


        if (sum_r[1]!= 1024*10)
        {
            std::cout << "Verification failed"<< std::endl;
        }

        auto kernel_offload_time = time.duration();
        std::cout << "Total time taken for the execution of nd_range parallel construct with "<< dim <<" dim \n and buff and acc is "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    
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
        

        time.start_timer();

        for ( i = 0; i < 10; i++)
        {
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
        }
        

        time.end_timer();

        auto sum_r = sum_buff.get_access<sycl::access::mode::read>();


        if (sum_r[1]!= 1024*10)
        {
            std::cout << "Verification failed"<< std::endl;
        }

        auto kernel_offload_time = time.duration();
        std::cout << "Total time taken for the execution of nd_range parallel construct with "<< dim <<" dim \n and buff and acc is "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    
    }
    else
    {
        std::cout << "ERROR: the dimension input should be 1 or 2 " << std::endl;
    }

    free(sum);
    
}


// reduction 

void reduction_with_atomics_buf_acc(sycl::queue &Q, int size)
{
    timer time;

    size = size*size;

    auto m = (TYPE *)malloc(size*sizeof(TYPE)); 
    std::fill(m,m+size,1.0);
    auto sum = (TYPE *)malloc(1*sizeof(TYPE)); 

    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};

    sycl::buffer<TYPE , 1> m_buff(m,size);

    sycl::buffer<TYPE , 1> sum_buff(sum,1);
        

    int i;

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {

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
    }   

    time.end_timer();

    auto sum_r = sum_buff.get_access<sycl::access::mode::read>();


    if (sum_r[0]!= size*10)
    {
        std::cout << "Verification failed"<< std::endl;
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for reduction using atomics and buffer and accessor memory management  "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    
    
    free(m);
    free(sum);
}

void reduction_with_atomics_usm(sycl::queue &Q, int size)
{
    timer time;

    size = size*size;

    auto m_shared = sycl::malloc_shared<TYPE>(size*sizeof(TYPE),Q); Q.wait();
    std::fill(m_shared,m_shared+size,1.0);
    auto sum = sycl::malloc_shared<TYPE>(1*sizeof(TYPE),Q); Q.wait();

    auto N = static_cast<size_t>(size);
    sycl::range<1> global{N};
    
    

    int i;

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {
        Q.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            auto j = it.get_id(0);

            auto v = sycl::atomic_ref<TYPE, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device,
                                  sycl::access::address_space::global_space>(
            sum[0]);

            
            v.fetch_add(m_shared[j]);
            
        });

        Q.wait();
    }   

    time.end_timer();


    if (sum[0]!= size*10)
    {
        std::cout << "Verification failed"<< std::endl;
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for reduction using atomics and USM memory management  "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    
    
    free(m_shared,Q);
}

#if defined(REDUCTION_IN_SYCL) 

void reduction_with_buf_acc(sycl::queue &Q, int size, int block_size)
{
    timer time;

    auto m_shared = (TYPE *)malloc(size*sizeof(TYPE));
    std::fill(m_shared,m_shared+size,1.0);
    auto sum = sycl::malloc_shared<TYPE>(1*sizeof(TYPE),Q); Q.wait();

    auto N = static_cast<size_t>(size);
    auto N_b = static_cast<size_t>(block_size);

    sycl::range<1> global{N};
    sycl::range<1> local{N_b};

    sycl::buffer<TYPE , 1> m_buff(m_shared,size);

    sycl::buffer<TYPE , 1> sum_buff(sum,1);
        

    int i;

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {

        Q.submit([&](sycl::handler& cgh){
            
            auto m_acc = m_buff.get_access<sycl::access::mode::read>(cgh);

            auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

            auto sum_red = sycl::reduction(sum_buff, cgh,sycl::plus<TYPE>());

            //auto sum_red = sycl::reduction(sum_acc, sycl::plus<TYPE>());

            cgh.parallel_for<>(sycl::nd_range<1>(global,local), sum_red ,[=](sycl::nd_item<1>it, auto &sum){

                auto j = it.get_global_id(0);

                sum += m_acc[j];
                
            });
        });

        Q.wait();
    }   

    time.end_timer();

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for reduction using atomics and USM memory management  "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    
    
    free(m_shared);
}

#endif


void global_barrier_test_usm(sycl::queue &Q, int size, int block_size)
{

    std::cout<< "\n Local range of the <nd_range> construct is: "<< block_size << std::endl;

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

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {
        Q.parallel_for<class global_barrier_usm>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            auto k = it.get_global_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum[k]+= 1;
            }

            it.barrier();
        
        });
        Q.wait();
    }
    

    time.end_timer();

    if (sum[0]!= 1024*10)
    {
        std::cout << "Verification failed"<< std::endl;
    }


    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the global barrier scope with 1 dim \n while using USM "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;

    free(sum,Q);
    
    
}


void global_barrier_test_buff_acc(sycl::queue &Q, int size, int block_size)
{

    std::cout<< "\n Local range of the <nd_range> construct is: "<< block_size << std::endl;

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

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {

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
    }
    

    time.end_timer();

    auto sum_r = sum_buff.get_access<sycl::access::mode::read>();

    if (sum_r[0]!= 1024*10)
    {
        std::cout << "Verification failed"<< std::endl;
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the global barrier scope with 1 dim \n while using buffer and accessors "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;

    free(sum);
    
    
}


void local_barrier_test_usm(sycl::queue &Q, int size, int block_size)
{

    std::cout<< "\n Local range of the <nd_range> construct is: "<< block_size << std::endl;

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

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {
        Q.parallel_for<class local_barrier_usm>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            auto k = it.get_global_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum[k]+= 1;
            }


            it.barrier(sycl::access::fence_space::local_space);
        
        });
        Q.wait();
    }
    

    time.end_timer();

    if (sum[0]!= 1024*10)
    {
        std::cout << "Verification failed"<< std::endl;
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the local barrier scope with 1 dim \n while using USM "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    

    free(sum,Q);

}

void local_barrier_test_buff_acc(sycl::queue &Q, int size, int block_size)
{

    std::cout<< "\n Local range of the <nd_range> construct is: "<< block_size << std::endl;

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

    time.start_timer();

    for ( i = 0; i < 10; i++)
    {

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
    }
    

    time.end_timer();

    auto sum_r = sum_buff.get_access<sycl::access::mode::read>();

    if (sum_r[0]!= 1024*10)
    {
        std::cout << "Verification failed"<< std::endl;
    }

    auto kernel_offload_time = time.duration();
    std::cout << "Total time taken for the local barrier scope with 1 dim \n while using buffer and accessors "<< kernel_offload_time/(10*1E9) << " seconds\n" << std::endl;
    
    free(sum);

    
}