#include "kernel.hpp"

#include <functional>
#include <iostream>
#include <list>
#include <memory>

#include <cuda_runtime.h>

template <typename T>
using deleted_unique_ptr = std::unique_ptr<T[], std::function<void(T *)>>;

template <typename T> class Array {
public:
    Array() = default;
    Array(size_t nElements) : m_size(nElements) {
        void *tmp = nullptr;
        cudaMallocManaged(&tmp, sizeof(T) * nElements);
        m_data = deleted_unique_ptr<T>(reinterpret_cast<T *>(tmp), cudaFree);
    }

    Array(size_t nElements, T value) : Array(nElements) { setValue(value); }

    void setValue(T value) {
        for (size_t i = 0; i < m_size; ++i)
            m_data[i] = value;
    }

    Array(Array const &) = delete;
    Array &operator=(Array const &) = delete;

    Array(Array &&other)
        : m_data(std::move(other.m_data)), m_size(other.m_size) {
        other.m_size = 0;
    }

    Array &operator=(Array &&rhs) {
        if (this == &rhs)
            return *this;

        m_data = std::move(rhs.m_data);
        m_size = rhs.m_size;
        rhs.m_size = 0;
        return *this;
    }

    size_t nBytes() const { return m_size * sizeof(T); }

    size_t size() const { return m_size; }

    T *data() { return m_data.get(); }
    T const *data() const { return m_data.get(); }

private:
    deleted_unique_ptr<T> m_data = nullptr;
    size_t m_size = 0;
};

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

void oversubscribeTest(size_t nMBperBatch, size_t totalGB, bool verbose) {

    const size_t nElements = nMBperBatch / (sizeof(double) / 1024.0 / 1024.0);

    Array<double> out(nElements, 0.0);
    std::list<Array<double>> arrays;
    double currentGB = 0;

    double counter = 0;
    do {
        arrays.emplace_front(nElements, counter++);
        auto &a = arrays.front();
        currentGB += a.nBytes() / (1024.0 * 1024.0 * 1024.0);
        // setArray<<<256, 256>>>(a.data(), counter++, a.size());
    } while (currentGB < totalGB);

    std::cout << arrays.size() << " arrays :: " << currentGB << " GB"
              << std::endl;
    const size_t nRuns = 10;
    for (size_t run = 0; run < nRuns; ++run) {
        for (auto const &arr : arrays) {
            addArray<<<256, 256>>>(arr.data(), out.data(), out.size());
        }
        cudaDeviceSynchronize();
        if (verbose)
            std::cout << arrays.size() << " arrays :: " << currentGB
                      << " GB :: sum = " << *out.data() << std::endl;
        // out.setValue(0.0);
        setArray<<<256, 256>>>(out.data(), 0.0, out.size());
    }

    // char c;
    // std::cin >> c;
}