# cudaOversubscribe

Github Actions: [![CMake](https://github.com/nitro44x/cudaOversubscribe/actions/workflows/cmake.yml/badge.svg)](https://github.com/nitro44x/cudaOversubscribe/actions/workflows/cmake.yml)

[Documentation](https://nitro44x.github.io/cudaOversubscribe/)

A very basic cli app to demonstrate nVidia's oversubscription capability. Oversubscription
is only implemented for Linux. Windows drivers need to be put in TCC mode (untested) for it
to work.

See nVidia's documentation for more information ([link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-oversubscription))

# Building and running the demo

Dependencies:

* C++ 11 compiler (gcc-11 and VS2019 tested)
* nvcc (CUDA 11.4 tested)
* Graphics card with CUDA capability >= 6.x

Run the following commands:

    git clone https://github.com/nitro44x/cudaOversubscribe
    cd cudaOversubscribe
    mkdir build
    cd build
    cmake ..
    make
    ./oversub --help


# Example runs

## 5 GB (no oversubscription)

Notice the Total Size of Host To Device is approximately 5 GB, so the memory was transfered to the device
only once and there were not device to host transfers. 

Also, the addArray kernel had a range of runtimes of 897us - 42ms.

    nvprof ./oversub -t 5
    
    Batch Size [MB]: 100
    Total Size [GB]: 5
    ==485180== NVPROF is profiling process 485180, command: ./oversub -t 5
    52 arrays :: 5.07812 GB
    ==485180== Profiling application: ./oversub -t 5
    ==485180== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    GPU activities:   99.82%  1.70004s       520  3.2693ms  897.13us  42.015ms  void addArray<dou...
                        0.18%  3.0576ms        10  305.76us  298.40us  312.26us  void setArray<do...
        API calls:   79.56%  1.70138s        10  170.14ms  47.169ms  1.27568s  cudaDeviceSynchronize
                    12.13%  259.33ms        53  4.8930ms  62.267us  255.74ms  cudaMallocManaged
                        8.19%  175.13ms        53  3.3043ms  3.2676ms  3.6595ms  cudaFree
                        0.08%  1.7380ms       530  3.2790us  2.8170us  39.461us  cudaLaunchKernel
                        0.03%  598.02us         1  598.02us  598.02us  598.02us  cuDeviceTotalMem
                        0.01%  164.18us       101  1.6250us     184ns  67.056us  cuDeviceGetAttribute
                        0.00%  29.274us         1  29.274us  29.274us  29.274us  cuDeviceGetName
                        0.00%  8.6750us         1  8.6750us  8.6750us  8.6750us  cuDeviceGetPCIBusId
                        0.00%  6.7060us         2  3.3530us     262ns  6.4440us  cuDeviceGet
                        0.00%  1.6610us         3     553ns     321ns     864ns  cuDeviceGetCount
                        0.00%     396ns         1     396ns     396ns     396ns  cuDeviceGetUuid

    ==485180== Unified Memory profiling result:
    Device "NVIDIA GeForce GTX 1080 Ti (0)"
    Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
    260105  20.562KB  4.0000KB  996.00KB  5.100624GB  618.4509ms  Host To Device
        9482         -         -         -           -   1.360747s  Gpu page fault groups
    Total CPU Page faults: 15900

## 20 GB (oversubscribe on GTX 1080 Ti)

When attempting to operate on 20 GB of data (-t 20 cli option), the volume of data transfered to the 
device jumped to 202 GB and device to host jumped to 192 GB. This number matches the number of times
the experiment is run (10 times), so this equates to transferring all of the data once per iteration.

The runtimes of addArray reflect this increased volume memory transfers with a range
of 21ms - 61ms. Note that the batch size is the same, so the runtime of addArray is a 1:1
comparison.

    nvprof ./oversub -t 20
    
    Batch Size [MB]: 100
    Total Size [GB]: 20
    ==485501== NVPROF is profiling process 485501, command: ./oversub -t 20
    205 arrays :: 20.0195 GB
    ==485501== Profiling application: ./oversub -t 20
    ==485501== Profiling result:
                Type  Time(%)      Time     Calls       Avg       Min       Max  Name
    GPU activities:  100.00%  63.3508s      2050  30.903ms  21.757ms  61.107ms  void addArray<dou...
                        0.00%  3.0349ms        10  303.49us  299.14us  310.75us  void setArray<do...
        API calls:   98.58%  63.3473s        10  6.33473s  6.06840s  6.37734s  cudaDeviceSynchronize
                        0.95%  613.53ms       206  2.9783ms  2.5939ms  3.3228ms  cudaFree
                        0.45%  290.90ms       206  1.4121ms  63.998us  271.95ms  cudaMallocManaged
                        0.01%  7.4226ms      2060  3.6030us  2.9630us  48.927us  cudaLaunchKernel
                        0.00%  768.89us         1  768.89us  768.89us  768.89us  cuDeviceTotalMem
                        0.00%  199.25us       101  1.9720us     238ns  81.000us  cuDeviceGetAttribute
                        0.00%  40.542us         1  40.542us  40.542us  40.542us  cuDeviceGetName
                        0.00%  10.044us         1  10.044us  10.044us  10.044us  cuDeviceGetPCIBusId
                        0.00%  2.7470us         3     915ns     448ns  1.6320us  cuDeviceGetCount
                        0.00%  1.7320us         2     866ns     272ns  1.4600us  cuDeviceGet
                        0.00%     510ns         1     510ns     510ns     510ns  cuDeviceGetUuid

    ==485501== Unified Memory profiling result:
    Device "NVIDIA GeForce GTX 1080 Ti (0)"
    Count  Avg Size  Min Size  Max Size  Total Size  Total Time  Name
    10004043  21.184KB  4.0000KB  0.9961MB  202.1111GB  24.212916s  Host To Device
    98321  2.0000MB  2.0000MB  2.0000MB  192.0332GB  15.804261s  Device To Host
    386068         -         -         -           -  80.920819s  Gpu page fault groups
    Total CPU Page faults: 61800

## Summary

For you table oriented people :bowtie:

| Total GB | H2D [GB] | D2H [GB] | Avg Runtime [ms] |
|:---:|:---:|:---:|:---:|
| 5 | 5 | 0 | 3.3 |
| 20 | 202 | 192 | 30.9 |
