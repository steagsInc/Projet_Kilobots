cmake_minimum_required(VERSION 3.15)
project(untitled C)

set(CMAKE_C_STANDARD 99)

add_executable(untitled main.c perceptron.c perceptron.h Layer.c Layer.h matmul.c matmul.h)