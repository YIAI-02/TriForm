cmake_minimum_required(VERSION 3.14)
project(rmsnorm_bench C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build options consistent with typical AscendC samples.
set(RUN_MODE "sim" CACHE STRING "cpu|sim|npu")
set(SOC_VERSION "Ascend910B3" CACHE STRING "Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4")
set(ASCEND_CANN_PACKAGE_PATH "" CACHE PATH "Path to Ascend Toolkit (e.g. /usr/local/Ascend/ascend-toolkit/latest)")

# Output
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/out")

add_executable(rmsnorm_bench
    src/main.cpp
    src/rmsnorm_runner.cpp
)

install(TARGETS rmsnorm_bench RUNTIME DESTINATION bin)
