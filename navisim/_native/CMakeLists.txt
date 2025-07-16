cmake_minimum_required(VERSION 3.20)
project(MultithreadTest VERSION 0.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(test_MT 
	main.cpp
)

set_target_properties(test_MT PROPERTIES CUDA_ARCHITECTURES "70;75;86")

target_include_directories(test_MT PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})