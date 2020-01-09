cmake_minimum_required(VERSION 3.9)
project(behlul VERSION 0.1
        DESCRIPTION "Behlul is an Efficient High Level Useful Library"
        LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-O3")

include_directories(./libs/include)
include_directories(./layers)
include_directories(./utils)

set(LAYER_FILES layers/conv_layer.cpp
                layers/max_pool.cpp
                layers/relu.cpp
                layers/dense_layer.cpp
                layers/softmax.cpp
                layers/cross_entropy.cpp)

find_package(pybind11 REQUIRED PATHS ./libs)
pybind11_add_module(my_project
                    binds.cpp
                    ${LAYER_FILES})

set(MAIN_TARGET run_main)

add_executable(${MAIN_TARGET}
                main.cpp
                ${LAYER_FILES}
                utils/mnist.cpp)
target_compile_features(${MAIN_TARGET} PUBLIC cxx_std_17)

set(UNIT_TEST_TARGET unit_test_main)

add_executable(${UNIT_TEST_TARGET}
                test/test_main.cpp
                ${LAYER_FILES}
                test/test_max_pool.cpp
                test/test_conv_layer.cpp
                test/test_dense_layer.cpp
                test/test_softmax.cpp
                test/test_cross_entropy.cpp
                test/test_relu.cpp)

