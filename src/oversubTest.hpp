#pragma once

/**
 * @brief Oversubscription run params
 */
struct Params {
    size_t batchSizeMB = 100; ///< Size (in MB) of each array. This controls the
                              ///< resolution of the test
    size_t totalGB = 3; ///< Least mount of memory to allocate. Arrays will be
                        ///< allocated of size batchSizeMB until the total mount
                        ///< is larger than totalGB.
    size_t nIterations = 5; ///< Number of times to run the calculation
    bool verbose = false;   ///< Print verbose info (per iteration).
};


struct testDriver {
    /**
     * @brief Driver function for oversubscription test
     *
     * This test allocates enough arrays to meet the totalGB desired,
     * then adds them all together. If you request more memory than
     * the graphics card has, then it will force the driver to automatically
     * migrate data to and from the card.
     *
     * 0. For each iteration
     * 1. Allocate N arrays
     * 2. For each array in arrays: output += array
     * 3. output = 0 (broadcast)
     * 4. repeat.
     *
     * @param params run parameters
     */
    static void oversubscribeTest(Params params);
};