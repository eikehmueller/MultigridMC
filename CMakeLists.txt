cmake_minimum_required(VERSION 3.21)

project(MultigridMC)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undefined-var-template")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

if((${CMAKE_MAJOR_VERSION} GREATER_EQUAL 3) AND(${CMAKE_MINOR_VERSION} GREATER_EQUAL 24))
  cmake_policy(SET CMP0135 NEW)
endif()

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Define user options
option(USE_CHOLMOD "Use Cholmod for Cholesky factorisation?" ON)

# use thorough testing?
option(USE_THOROUGH_TESTS "Use thorough tests?" ON)

find_package(Eigen3 5.0 REQUIRED NO_MODULE)
include(FindEigenLAPACK)

# Find Cholmod
set(NCHOLMOD "true")

if(USE_CHOLMOD)
  include(FindCholMod)
endif(USE_CHOLMOD)

# Find libconfig library
include(FindLibConfig)

# Find OpenMP
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
  link_libraries(OpenMP::OpenMP_CXX)
  add_link_options(${OpenMP_CXX_FLAGS})
endif()

# Initialize global list of cc files
set(SOURCES "" CACHE INTERNAL "sources")

include_directories(src)

add_subdirectory(src)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
