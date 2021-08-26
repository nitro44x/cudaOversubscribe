#pragma once

struct Params {
    size_t batchSizeMB = 100;
    size_t totalGB = 3;
    size_t nIterations = 5;
    bool verbose = false;
};

void oversubscribeTest(Params params);