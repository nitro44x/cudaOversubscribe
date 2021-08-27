#include <CLI11.hpp>

#include "oversubTest.hpp"

#include <iostream>

int main(int argc, char **argv) {
    CLI::App app{"CUDA GPU oversubscription demo"};

    Params params;
    app.add_option("-s,--size,size", params.batchSizeMB, "Batch size [MB]")
        ->default_val(params.batchSizeMB);

    app.add_option("-t,--total,total", params.totalGB,
                   "Total amount of memory to allocate [GB]")
        ->default_val(params.totalGB);

    app.add_option("-n,--iterations,iterations", params.nIterations,
                   "Number of iterations to operate on the dataset")
        ->default_val(params.nIterations);

    auto verbose = app.add_flag("-v,--verbose", params.verbose);

    CLI11_PARSE(app, argc, argv);

    std::cout << std::endl;
    std::cout << "Batch Size [MB]: " << params.batchSizeMB << std::endl;
    std::cout << "Total Size [GB]: " << params.totalGB << std::endl;
    std::cout << "Iterations: " << params.nIterations << std::endl;
    std::cout << "Verbose: " << params.verbose << std::endl;
    std::cout << std::endl;

    testDriver::oversubscribeTest(params);

    return 0;
}