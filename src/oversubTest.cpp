#include "oversubTest.hpp"

#include "Array.hpp"

#include <iostream>
#include <list>

#include <cuda_runtime.h>

void testDriver::oversubscribeTest(Params params) {

    const size_t nElements =
        params.batchSizeMB / (sizeof(double) / 1024.0 / 1024.0);

    Array<double> out(nElements, 0.0);
    std::list<Array<double>> arrays;
    double currentGB = 0;

    double counter = 0;
    do {
        arrays.emplace_front(nElements, counter++);
        auto &a = arrays.front();
        currentGB += a.nBytes() / (1024.0 * 1024.0 * 1024.0);
    } while (currentGB < params.totalGB);

    std::cout << arrays.size() << " arrays :: " << currentGB << " GB"
              << std::endl;

    for (size_t run = 0; run < params.nIterations; ++run) {

        for (auto const &arr : arrays)
            out += arr;

        auto const err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error occurred (" << err
                      << "): " << cudaGetErrorString(err) << std::endl;
            return;
        }
        if (params.verbose)
            std::cout << run << ": sum = " << *out.data() << std::endl;

        out.setValue(0.0);
    }

    auto const err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error occurred (" << err
                  << "): " << cudaGetErrorString(err) << std::endl;
    }
}