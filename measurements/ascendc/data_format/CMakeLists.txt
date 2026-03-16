cmake_minimum_required(VERSION 3.16)
project(ascend_format_conv_bench)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend910B1" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/format_conv_bench.cpp)
set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message(FATAL_ERROR "invalid RUN_MODE: ${RUN_MODE}")
endif()

add_executable(format_conv_bench_app
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_compile_options(format_conv_bench_app PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_link_libraries(format_conv_bench_app PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE}
    tiling_api
    register
    platform
    ascendalog
    dl
)

install(TARGETS format_conv_bench_app
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
