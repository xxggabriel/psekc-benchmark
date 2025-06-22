cmake_minimum_required(VERSION 3.18)
project(PseKNC_Benchmark LANGUAGES CXX CUDA)

# Permite escolher o tipo de build (Debug, Release, RelWithDebInfo)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

message(STATUS "Procurando dependências...")
find_package(CURL REQUIRED)
message(STATUS " - CURL encontrado: ${CURL_LIBRARIES}")
find_package(Threads REQUIRED)
message(STATUS " - Threads encontrado.")
find_package(OpenMP REQUIRED)
message(STATUS " - OpenMP encontrado.")

if(CMAKE_CUDA_COMPILER)
    message(STATUS " - Compilador CUDA encontrado: ${CMAKE_CUDA_COMPILER}")
    set(CUDA_SUPPORT_ENABLED TRUE)
else()
    message(WARNING " - Compilador CUDA NÃO encontrado. O código da GPU será ignorado.")
    set(CUDA_SUPPORT_ENABLED FALSE)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE   "-O3 -DNDEBUG -march=x86-64" CACHE STRING "Flags for Release build" FORCE)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO   "-O2 -g" CACHE STRING "Flags for RelWithDebInfo build" FORCE)
    set(CMAKE_CXX_FLAGS_DEBUG     "-g" CACHE STRING "Flags for Debug build" FORCE)
    message(STATUS "Flags C++ definidas para GCC/Clang.")
endif()

if(CUDA_SUPPORT_ENABLED)

    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -gencode arch=compute_61,code=sm_61")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELWITHDEBINFO} -lineinfo -g")
    message(STATUS "Flags CUDA definidas para Debug/RelWithDebInfo.")
endif()

add_executable(benchmark
        src/include/DataManager.h
        src/include/DatasetGenerator.h
        src/include/PseKNCProcessor.h
        src/include/CPUProcessor.h
        src/include/OMPProcessor.h
        src/include/BenchmarkRunner.h
        src/include/ReportGenerator.h
        src/main.cpp
        src/DataManager.cpp
        src/DatasetGenerator.cpp
        src/PseKNCProcessor.cpp
        src/CPUProcessor.cpp
        src/OMPProcessor.cpp
        src/BenchmarkRunner.cpp
        src/ReportGenerator.cpp
)

target_include_directories(benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# GPU
if(CUDA_SUPPORT_ENABLED)
    target_sources(benchmark PRIVATE
            src/include/GPUProcessor.h
            src/GPUProcessor.cu
    )
    target_compile_definitions(benchmark PRIVATE WITH_CUDA)
    # Define propriedades CUDA
    set_target_properties(benchmark PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES 61
    )
endif()

target_link_libraries(benchmark PRIVATE
        CURL::libcurl
        Threads::Threads
        OpenMP::OpenMP_CXX
)

set_target_properties(benchmark PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED YES
)

message(STATUS "Configuração do CMake concluída.")
