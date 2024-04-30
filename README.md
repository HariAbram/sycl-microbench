# sycl-microbench

This is a micro-benchmark for testing the overhead of SYCL features, the following features are tested in this benchmark 

* memory allocation
* parallelization 
* atomics 
* barriers
* vectorization

# Building 

CMake is used to build this benchmark. 

```
mkdir build && cd build
cmake .. -DSYCL_COMPILE= DPCPP|HIPSYCL -DOMP_COMPILE=true|false
make
```
if `HIPSYCL` is chosen as a SYCL implementation then `-DHIPSYCL_INSTALL_DIR` need to be specified. 

For optimal performance `OMP_PROC_BIND` environment variable is set to true. 

### Example

```
./binary " [-s size |-b blocksize <optional>|\n
   	 --mat-mul : to run matrix multiplication \n
	 --vec-add : to run vector addition \n
	   can run only mat-mul or vec-add at a time, can't run both simultaneously \n
	 --mem-alloc : to alloc memory using SYCL and standard malloc \n
	 --reduction : to test reduction using atomics and sycl reduction construct
	 --range : to test sycl range construct
	 --ndrange : to test sycl nd_range construct
	 -i : for different routines in vectorization benchmark\n
	       1 - range with USM\n
	       2 - range with Buffer and Accessors\n
	       3 - nd_range with USM\n
	       4 - nd_range with Buffer and Accessor\n
   
```

