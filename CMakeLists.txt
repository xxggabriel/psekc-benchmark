cmake_minimum_required(VERSION 3.18)
project(PseKNC_Benchmark LANGUAGES CXX CUDA)


find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

    
add_executable(benchmark
    src/main.cpp
    src/DataManager.cpp
    src/CPUProcessor.cpp
    src/OMPProcessor.cpp
    src/GPUProcessor.cu
    src/include/DataManager.h
    src/include/PseKNCProcessor.h
    src/include/CPUProcessor.h
    src/include/OMPProcessor.h
    src/include/GPUProcessor.h
)


target_include_directories(benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)


target_link_libraries(benchmark
    PRIVATE
    CURL::libcurl
    Threads::Threads
)


set_target_properties(benchmark PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
)