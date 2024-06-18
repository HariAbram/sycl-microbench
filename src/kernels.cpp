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

void init_arrays(sycl::queue &Q, sycl::_V1::buffer<TYPE, 1>  m_buff, sycl::_V1::buffer<TYPE, 1> a_buff, sycl::range<1> global)
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

void kernel_copy(sycl::queue &Q, sycl::_V1::buffer<TYPE, 1> m_buff, sycl::_V1::buffer<TYPE, 1> a_buff, sycl::range<1> global)
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

