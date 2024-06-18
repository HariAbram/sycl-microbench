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

using namespace cl;

/////////////////////////////////////////////// init arrays

void init_arrays(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1> it){

            const int k = it.get_id(0);

            m[k] = 0.0;

            a[k] = 1.0;

        });
    });
    Q.wait();

}

void init_arrays(sycl::queue &Q, sycl::buffer<TYPE, 1>  m_buff, sycl::buffer<TYPE, 1> a_buff, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){

        sycl::accessor m_acc(m_buff, cgh, sycl::write_only, sycl::no_init);
        sycl::accessor a_acc(a_buff, cgh, sycl::write_only, sycl::no_init);

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int k = it.get_id(0);

            m_acc[k] = 0.0 ;
            a_acc[k] = 1.0;
        
        });

    });
    Q.wait();

}

/////////////////////////////////////////////// copy

void kernel_copy(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1> it){

            const int k = it.get_id(0);

            m[k] = a[k];

        });
    });
    Q.wait();

}

void kernel_copy(sycl::queue &Q, TYPE *m, TYPE *a, sycl::range<1> global, sycl::range<1> local)
{
    Q.submit([&](sycl::handler& cgh){

        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1> it){

            const int k = it.get_global_id(0);

            m[k] = a[k];

        });
    });
    Q.wait();

}

void kernel_copy(sycl::queue &Q, sycl::buffer<TYPE, 1> m_buff, sycl::buffer<TYPE, 1> a_buff, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){

        auto m_acc = m_buff.get_access<sycl::access::mode::write>(cgh);
        auto a_acc = a_buff.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            const int k = it.get_id(0);

            m_acc[k] = a_acc[k];
        
        });

    });

}

void kernel_copy(TYPE* m, TYPE* a, int size)
{
    #pragma omp parallel for 
    for (size_t j = 0; j < size*size; j++)
    {
    m[j] = a[j];
    }

}

/////////////////////////////////////////////// parallel 
/// 1 dim

void kernel_parallel_1(sycl::queue &Q, TYPE* sum, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){

            auto k = it.get_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum[k] += 1;
            }
        });
    });

    Q.wait();

}

void kernel_parallel_1(sycl::queue &Q, TYPE* sum, sycl::range<1> global, sycl::range<1> local)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){

            auto k = it.get_global_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum[k] += 1;    
            }
        });
    });
    Q.wait();
}


void kernel_parallel_1(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<1> global)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::range<1>(global), [=](sycl::item<1>it){
            
            auto k = it.get_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum_acc[k] += 1; 
            }
        });
    });
    Q.wait();
}

void kernel_parallel_1(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<1> global, sycl::range<1> local)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::nd_range<1>(global,local), [=](sycl::nd_item<1>it){
            
            auto k = it.get_global_id(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum_acc[k] += 1;    
            }
        });
    });
    Q.wait();

}

/// 2 dim
void kernel_parallel_2(sycl::queue &Q, TYPE* sum, sycl::range<2> global)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::range<2>(global), [=](sycl::item<2>it){

            auto k = it.get_id(0);
            auto k1 = it.get_id(1);

            auto N = it.get_range(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum[k*N+k1] += 1;  
            }
        });
    });

    Q.wait();

}

void kernel_parallel_2(sycl::queue &Q, TYPE* sum, sycl::range<2> global, sycl::range<2> local)
{
    Q.submit([&](sycl::handler& cgh){
        cgh.parallel_for<>(sycl::nd_range<2>(global,local), [=](sycl::item<2>it){

            auto k = it.get_id(0);
            auto k1 = it.get_id(1);

            auto N = it.get_range(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum[k*N+k1] += 1;    
            }
        });
    });
    Q.wait();
}

void kernel_parallel_2(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<2> global)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::range<2>(global), [=](sycl::item<2>it){

            auto k = it.get_id(0);
            auto k1 = it.get_id(1);

            auto N = it.get_range(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum_acc[k*N+k1] += 1;     
            }
        });
    });
    Q.wait();
}

void kernel_parallel_2(sycl::queue &Q, sycl::buffer<TYPE, 1> sum_buff, sycl::range<2> global, sycl::range<2> local)
{
    Q.submit([&](sycl::handler& cgh){
        auto sum_acc = sum_buff.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<>(sycl::nd_range<2>(global,local), [=](sycl::item<2>it){

            auto k = it.get_id(0);
            auto k1 = it.get_id(1);

            auto N = it.get_range(0);

            for (size_t l = 0; l < 1024; l++)
            {
                sum_acc[k*N+k1] += 1;
            }        
        });
    });
    Q.wait();
}