#include "include/GPUProcessor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <map>

#define CUDA_CHECK(err) {                                   \
    cudaError_t e = (err);                                  \
    if (e != cudaSuccess) {                                 \
        fprintf(stderr, "Erro CUDA em %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
}

__device__ int d_seq_counter = 0;

__global__ void pseknc_dyn_kernel(
    const int *__restrict__ indices,
    const int *__restrict__ offsets,
    int num_sequences,
    const double *__restrict__ prop_table,
    int lambda_max,
    double weight,
    int num_sorted_ktuples,
    int num_properties,
    int vec_dim,
    double *__restrict__ out) {
    // A memória é alocada como um bloco de bytes (char) para garantir o alinhamento correto.
    extern __shared__ char s_mem[];
    int *s_counts = (int *) s_mem;
    // s_reduction agora aponta para a posição de memória DEPOIS do array s_counts.
    double *s_reduction = (double *) (&s_counts[num_sorted_ktuples]);

    while (true) {
        __shared__ int seq_idx_shared;
        if (threadIdx.x == 0) {
            seq_idx_shared = atomicAdd(&d_seq_counter, 1);
        }
        __syncthreads();

        int seq_idx = seq_idx_shared;
        if (seq_idx >= num_sequences) {
            break;
        }

        int start_offset = offsets[seq_idx];
        int end_offset = offsets[seq_idx + 1];
        int num_k_tuples = end_offset - start_offset;

        if (num_k_tuples <= 0) {
            if (threadIdx.x == 0) {
                double *outv = out + (long long) seq_idx * vec_dim;
                for (int i = 0; i < vec_dim; ++i) outv[i] = 0.0;
            }
            continue;
        }

        for (int i = threadIdx.x; i < num_sorted_ktuples; i += blockDim.x) s_counts[i] = 0;
        __syncthreads();

        for (int i = threadIdx.x; i < num_k_tuples; i += blockDim.x) {
            atomicAdd(&s_counts[indices[start_offset + i]], 1);
        }
        __syncthreads();

        double correlation_factors[32];

        for (int l = 1; l <= lambda_max; ++l) {
            long long num_pairs = (long long) num_k_tuples - l;
            double thread_sum = 0.0;

            if (num_pairs > 0) {
                for (long long i = threadIdx.x; i < num_pairs; i += blockDim.x) {
                    int idx1 = indices[start_offset + i];
                    int idx2 = indices[start_offset + i + l];
                    double sum_sq_diff = 0.0;
                    for (int p = 0; p < num_properties; ++p) {
                        double d = prop_table[(long long) idx1 * num_properties + p] -
                                   prop_table[(long long) idx2 * num_properties + p];
                        sum_sq_diff += d * d;
                    }
                    thread_sum += sum_sq_diff / num_properties;
                }
            }

            s_reduction[threadIdx.x] = thread_sum;
            __syncthreads();
            for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
                if (threadIdx.x < offset) {
                    s_reduction[threadIdx.x] += s_reduction[threadIdx.x + offset];
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                correlation_factors[l - 1] = (num_pairs > 0) ? (s_reduction[0] / num_pairs) : 0.0;
            }
        }

        if (threadIdx.x == 0) {
            double correlation_sum = 0.0;
            for (int i = 0; i < lambda_max; ++i) {
                correlation_sum += correlation_factors[i];
            }

            double denominator = 1.0 + weight * correlation_sum;
            double *outv = out + (long long) seq_idx * vec_dim;

            for (int i = 0; i < num_sorted_ktuples; ++i) {
                outv[i] = (static_cast<double>(s_counts[i]) / num_k_tuples) / denominator;
            }
            for (int i = 0; i < lambda_max; ++i) {
                outv[num_sorted_ktuples + i] = (weight * correlation_factors[i]) / denominator;
            }
        }
    }
}

GPUProcessor::GPUProcessor(const PropertiesMap &properties, PseKNCParams params)
    : PseKNCProcessor(properties, params) {
    setup_gpu_data();
}

GPUProcessor::~GPUProcessor() {
    if (d_property_table) {
        CUDA_CHECK(cudaFree(d_property_table));
    }
}

void GPUProcessor::setup_gpu_data() {
    int idx = 0;
    for (const auto &kt: sorted_ktuples) {
        k_tuple_to_index[kt] = idx++;
    }

    num_properties = properties_map.begin()->second.size();
    int K = static_cast<int>(sorted_ktuples.size());
    size_t bytes = static_cast<size_t>(K) * num_properties * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_property_table, bytes));
    std::vector<double> host_buf(K * num_properties);
    for (const auto &kt: sorted_ktuples) {
        int row = k_tuple_to_index.at(kt);
        const auto &props = properties_map.at(kt);
        for (size_t c = 0; c < num_properties; ++c) {
            host_buf[row * num_properties + c] = props[c];
        }
    }
    CUDA_CHECK(cudaMemcpy(d_property_table,
        host_buf.data(),
        bytes,
        cudaMemcpyHostToDevice));
}

std::vector<double> GPUProcessor::process(const std::string &sequence) {
    throw std::runtime_error("GPUProcessor::process não implementado; use process_batch.");
}

std::vector<std::vector<double> > GPUProcessor::process_batch(
    const std::vector<std::string> &sequences) {
    int N = static_cast<int>(sequences.size());

    std::vector<int> h_idx;
    std::vector<int> h_off(N + 1);
    h_off[0] = 0;
    for (int i = 0; i < N; ++i) {
        auto kt = prepare_ktuples(sequences[i]);
        for (const auto &t: kt) {
            h_idx.push_back(k_tuple_to_index.at(t));
        }
        h_off[i + 1] = static_cast<int>(h_idx.size());
    }

    int K = static_cast<int>(sorted_ktuples.size());
    int L = params.lambda_max;
    int vec_dim = K + L;

    int *d_idx = nullptr;
    int *d_off = nullptr;
    double *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_idx, h_idx.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_off, h_off.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<long long>(N) * vec_dim * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_idx, h_idx.data(), h_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_off, h_off.data(), h_off.size() * sizeof(int), cudaMemcpyHostToDevice));

    int zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_seq_counter, &zero, sizeof(int)));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int blocks = prop.multiProcessorCount * 8; // Aumenta a ocupação
    int threads_per_block = 128;

    size_t shared_bytes = static_cast<size_t>(K) * sizeof(int) + threads_per_block * sizeof(double);


    pseknc_dyn_kernel<<<blocks, threads_per_block, shared_bytes>>>(
        d_idx, d_off, N,
        d_property_table,
        L, params.weight,
        K, num_properties,
        vec_dim,
        d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> host_out(static_cast<long long>(N) * vec_dim);
    CUDA_CHECK(cudaMemcpy(host_out.data(), d_out, host_out.size() * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_idx));
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_out));

    std::vector<std::vector<double> > result(N);
    for (int i = 0; i < N; ++i) {
        result[i].assign(
            host_out.begin() + static_cast<long long>(i) * vec_dim,
            host_out.begin() + static_cast<long long>(i + 1) * vec_dim);
    }
    return result;
}
