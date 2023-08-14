# sycl-microbench

This is a micro-benchmark for testing overhead of SYCL features, the following features are tested in this benchmark 

* memory allocation
* parallelization 
* atomics 
* barriers

# Building 

CMake is used to build this benchmark. 

```
mkdir build && cd build
cmake .. -DSYCL_COMPILE= DPCPP|HIPSYCL -DOMP_COMPILE=true|false
```
if `HIPSYCL` is specifed as a SYCL implementation then `-DHIPSYCL_INSTALL_DIR` need to be specified. Similary, when `OMP_COMPILE` is true then 'OMP_LIBRARY' need to be specified. 

For optimal performance `OMP_PROC_BIND` is set to true. 

### Example

```
./binary 
   -s <problem size>
   -b <blovk size (optional)>
```

