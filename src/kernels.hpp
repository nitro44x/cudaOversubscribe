
/**
 * @file kernels.hpp
 * @brief Simple helper kernels
 */

#pragma once

#include <cuda_runtime.h>

namespace kernels {

/**
 * @brief CUDA kernel that performs an inplace add
 *
 * @tparam T data type (must support +)
 * @param a array to add
 * @param out array the inplace addition is performed on
 * @param N size of a and out
 */
template <typename T> __global__ void addArray(T const *a, T *out, size_t N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        out[tid] += a[tid];
        tid += blockDim.x * gridDim.x;
    }
}

/**
 * @brief CUDA kernel that sets an entire array to a single value.
 *
 * @tparam T data type
 * @param a array to be set
 * @param value the value to set the array too
 * @param N size of the array
 */
template <typename T> __global__ void setArray(T *a, T value, size_t N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        a[tid] = value;
        tid += blockDim.x * gridDim.x;
    }
}
} // namespace kernels
