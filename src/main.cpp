#include "CLI11.hpp"
#include "kernel.hpp"

#include <iostream>

int main(int argc, char **argv) {
    CLI::App app{"CUDA GPU oversubscription demo"};

    size_t batchSizeMB = 100;
    app.add_option("-s,--size,size", batchSizeMB, "Batch size [MB]")
        ->default_val(batchSizeMB);

    size_t totalGB = 3;
    app.add_option("-t,--total,total", totalGB,
                   "Total amount of memory to allocate [GB]")
        ->default_val(totalGB);

    auto verbose = app.add_flag("-v,--verbose");

    CLI11_PARSE(app, argc, argv);

    std::cout << "Batch Size [MB]: " << batchSizeMB << std::endl;
    std::cout << "Total Size [GB]: " << totalGB << std::endl;

    oversubscribeTest(batchSizeMB, totalGB, verbose->count());

    return 0;
}