cmake_minimum_required(VERSION 3.18)
project(PseKNC_Benchmark LANGUAGES CXX CUDA)


message(STATUS "A procurar dependências...")
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

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG -march=x86-64" CACHE STRING "Flags for Release build" FORCE)
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG -march=x86-64" CACHE STRING "Flags for Release build" FORCE)
    message(STATUS "Flags de compilação para GCC/Clang definidas para portabilidade (-march=x86-64).")
endif()


add_executable(benchmark
        src/include/DataManager.h
        src/include/PseKNCProcessor.h
        src/include/CPUProcessor.h
        src/include/OMPProcessor.h
        src/main.cpp
        src/DataManager.cpp
        src/PseKNCProcessor.cpp
        src/CPUProcessor.cpp
        src/OMPProcessor.cpp
        src/include/BenchmarkRunner.h
        src/BenchmarkRunner.cpp
        src/include/ReportGenerator.h
        src/ReportGenerator.cpp
)

if(CUDA_SUPPORT_ENABLED)
    target_sources(benchmark PRIVATE
            src/include/GPUProcessor.h
            src/GPUProcessor.cu
    )
    target_compile_definitions(benchmark PRIVATE WITH_CUDA)
endif()

target_include_directories(benchmark PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(benchmark PRIVATE CURL::libcurl Threads::Threads OpenMP::OpenMP_CXX)
set_target_properties(benchmark PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED YES)
message(STATUS "Configuração do CMake concluída.")