# Intall mlirfuzzer-opt tools

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX`. To build and launch the tests, run
```sh
$ export BUILD_DIR=/path/compiler-testing/external/llvm/build
$ cd MLIRFuzzer
$ mkdir build && cd build
$ cmake -G Ninja .. \
    -DMLIR_DIR=/data/mlirfuzzer/external/llvm/build/lib/cmake/mlir \
    -DCMAKE_C_COMPILER=/data/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04/bin/clang \
    -DCMAKE_CXX_COMPILER=/data/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04/bin/clang++
$ ninja


$ mkdir /home/MLIRFuzzer && cd /home/MLIRFuzzer
$ mkdir build && cd build
$ /data/cmake-3.22.0-linux-x86_64/bin/cmake -G Ninja /data/mlirfuzzer/MLIRFuzzer/ \
    -DMLIR_DIR=/data/mlirfuzzer/external/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=/data/mlirfuzzer/external/llvm/build/lib/cmake/llvm \
    -DCMAKE_C_COMPILER=/data/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang \
    -DCMAKE_CXX_COMPILER=/data/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04/bin/clang++
$ ninja
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

