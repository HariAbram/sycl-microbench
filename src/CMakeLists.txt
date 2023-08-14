cmake_minimum_required(VERSION 3.14)

if(${SYCL_RUNTIME} STREQUAL "DPCPP")
   if (NOT DPCPP_BIN)
      message(STATUS "DPCPP_BIN unspecified, using dpcpp on path")
      set(CMAKE_CXX_COMPILER "dpcpp")
   else()
      set(CMAKE_CXX_COMPILER ${DPCPP_BIN})
  endif ()
  
endif()

project(micro-bench-opensycl LANGUAGES CXX C)

include_directories(${CMAKE_SOURCE_DIR}/../include)

set(benchmarks
  ./microbench.cpp
)



if(SYCL_RUNTIME)
  
  if (${SYCL_RUNTIME} STREQUAL "HIPSYCL")

        if (NOT HIPSYCL_INSTALL_DIR)
            message(FATAL_ERROR "HIPSYCL_INSTALL_DIR is undefined")
        endif ()
        set(hipSYCL_DIR ${HIPSYCL_INSTALL_DIR}/lib/cmake/hipSYCL)

        if (NOT EXISTS "${hipSYCL_DIR}")
            message(WARNING "Falling back to hipSYCL < 0.9.0 CMake structure")
            set(hipSYCL_DIR ${HIPSYCL_INSTALL_DIR}/lib/cmake)
        endif ()
        if (NOT EXISTS "${hipSYCL_DIR}")
            message(FATAL_ERROR "Can't find the appropriate CMake definitions for hipSYCL")
        endif ()

        find_package(hipSYCL CONFIG REQUIRED)
     
  elseif (${SYCL_RUNTIME} STREQUAL "DPCPP")

        if (NOT DPCPP_BIN)
            message(STATUS "DPCPP_BIN unspecified, using dpcpp on path")
            set(CMAKE_CXX_COMPILER "dpcpp")
        else()
            set(CMAKE_CXX_COMPILER ${DPCPP_BIN})
        endif ()

        if(DPCPP_INCLUDE)
            include_directories(${DPCPP_INCLUDE})
        endif()
        

        add_definitions(-DCL_TARGET_OPENCL_VERSION=220)
        set(EXTRA_FLAGS -pedantic)
    else ()
        message(FATAL_ERROR "SYCL_RUNTIME unsupported, must be one of HIPSYCL|DPCPP, got ${SYCL_RUNTIME}")
    endif ()
  
  
  find_package(Threads REQUIRED)
  
  
  foreach(benchmark IN LISTS benchmarks)
    get_filename_component(target ${benchmark} NAME_WE)

    add_executable(${target} ${benchmark} ./timer.cpp)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin/")
    
    target_link_libraries(${target} 
    PRIVATE 
      Threads::Threads
    ) 
    
    target_compile_options(${target}
    PRIVATE
      -O3
      -g
    )
    
    if(${SYCL_RUNTIME} STREQUAL "HIPSYCL")
      add_sycl_to_target(TARGET ${target} SOURCES ${benchmark} ./timer.cpp)
    endif()

    install(TARGETS ${target} RUNTIME DESTINATION bin/)
    get_filename_component(dir ${benchmark} DIRECTORY)
    set_property(TARGET ${target} PROPERTY FOLDER ${dir})
  endforeach(benchmark)

else() 

  message(FATAL_ERROR "SYCL_RUNTIME not specified.")

endif(SYCL_RUNTIME)

if(OMP_COMPILE)

  add_executable(micro-bench-omp ./micro-bench-omp.cpp )
  set_target_properties(micro-bench-omp PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")

  target_compile_options(micro-bench-omp
  PRIVATE
    -O3
    -g
    -fopenmp
  )

  target_compile_features(micro-bench-omp
  PRIVATE
    cxx_std_11
  )

  find_package(Threads REQUIRED)
    # ...and link against it
    
  if(OMP_LIBRARY)
    target_link_libraries(micro-bench-omp
      PRIVATE 
        Threads::Threads
        ${OMP_LIBRARY}
    )
  else()
    message(FATAL_ERROR "OMP LIBRARY NOT SPECIFIED")
  endif(OMP_LIBRARY)

endif(OMP_COMPILE)

