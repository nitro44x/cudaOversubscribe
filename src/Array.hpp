#pragma once

#include <functional>
#include <memory>

#include <cuda_runtime.h>

namespace kernels {
template <typename T> __global__ void addArray(T const *a, T *out, size_t N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        out[tid] += a[tid];
        tid += blockDim.x * gridDim.x;
    }
}

template <typename T> __global__ void setArray(T *a, T value, size_t N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < N) {
        a[tid] = value;
        tid += blockDim.x * gridDim.x;
    }
}
} // namespace kernels

template <typename T>
using deleted_unique_ptr = std::unique_ptr<T[], std::function<void(T *)>>;

/**
 * @brief Basic array class to handle CUDA memory management.
 *
 * @tparam T element type of the array, must be a numeric type.
 */
template <typename T> class Array {
public:
    Array() = default;

    /**
     * @brief Construct a new Array object
     *
     * @param nElements number of elements to allocate.
     */
    Array(size_t nElements) : m_size(nElements) {
        void *tmp = nullptr;
        cudaMallocManaged(&tmp, sizeof(T) * nElements);
        m_data = deleted_unique_ptr<T>(reinterpret_cast<T *>(tmp), cudaFree);
    }

    /**
     * @brief Construct a new Array object
     *
     * @param nElements number of elements to allocate
     * @param value Value to initialize each element too.
     */
    Array(size_t nElements, T value) : Array(nElements) { setValue(value); }

    /**
     * @brief Set the value of each element in the array
     */
    void setValue(T value) {
        kernels::setArray<<<256, 256>>>(data(), value, size());
    }

    Array(Array const &) = delete;
    Array &operator=(Array const &) = delete;

    /**
     * @brief Copy constructor.
     */
    Array(Array &&other)
        : m_data(std::move(other.m_data)), m_size(other.m_size) {
        other.m_size = 0;
    }

    /**
     * @brief Copy assign operator.
     */
    Array &operator=(Array &&rhs) {
        if (this == &rhs)
            return *this;

        m_data = std::move(rhs.m_data);
        m_size = rhs.m_size;
        rhs.m_size = 0;
        return *this;
    }

    /**
     * @brief Inplace += executed on the GPU
     *
     * @param rhs Array to add into the current array
     * @return Array& The inplace array
     */
    Array &operator+=(Array<T> const &rhs) {
        kernels::addArray<<<256, 256>>>(rhs.data(), data(), size());
        return *this;
    }

    /**
     * @brief Number of bytes used by this array
     */
    size_t nBytes() const { return m_size * sizeof(T); }

    /**
     * @brief Number of elements in the array
     */
    size_t size() const { return m_size; }

    /**
     * @brief Get underlying data pointer.
     */
    T *data() { return m_data.get(); }

    /**
     * @brief Get underlying const data pointer.
     */
    T const *data() const { return m_data.get(); }

private:
    deleted_unique_ptr<T> m_data = nullptr;
    size_t m_size = 0;
};
