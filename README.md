# sycl-microbench

This is a micro-benchmark for testing overhead of SYCL featuers, the following features are tested in this benchmark 

* memory allocation
* parallelization 
* atomics 
* barriers

# Building 

CMake is used to build this benchmark. 

`mkdir build && cd build`
`cmake .. -DSYCL_COMPILE= DPCPP|HIPSYCL -DOMP_COMPILE=true|false`

