#include <iostream>

#include "kernel.hpp"

int main() { 
    std::cout << "3 + 4 = " << add(3, 4) << std::endl;

    // 15gb oversubscribes a GTX 1080 Ti (11gb)
    oversubscribeTest(100, 15);

    // 5gb does not oversub
    //oversubscribeTest(100, 5);

    return 0;
}