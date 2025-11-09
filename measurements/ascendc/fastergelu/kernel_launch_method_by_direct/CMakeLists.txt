cmake_minimum_required(VERSION 3.16)
project(Ascend_c)
if(${RUN_MODE})
    set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
endif()
if (${SOC_VERSION})
    set(SOC_VERSION "Ascend910" CACHE STRING "system on chip type")
endif()

set(ASCEND_CANN_PACKAGE_PATH "~/Ascend/ascend-toolkit/latest" CACHE STRING "ASCEND CANN package installation directory")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug"  CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out"  CACHE STRING "path for install()" FORCE)
endif()

file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/faster_gelu_custom.cpp
)
set(CUSTOM_ASCEND310P_LIST "Ascend310P1" "Ascend310P3")

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message("invalid RUN_MODE: ${RUN_MODE}")
endif()

add_executable(fastergelu_direct_kernel_op
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/faster_gelu_custom_tiling.cpp
)

target_compile_options(fastergelu_direct_kernel_op PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2
    -std=c++17
    -D_GLIBCXX_USE_CXX11_ABI=0
)

target_compile_definitions(fastergelu_direct_kernel_op PRIVATE
    $<$<BOOL:$<IN_LIST:${SOC_VERSION},${CUSTOM_ASCEND310P_LIST}>>:CUSTOM_ASCEND310P>
)

target_include_directories(fastergelu_direct_kernel_op PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:${ASCEND_CANN_PACKAGE_PATH}/include>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:${ASCEND_CANN_PACKAGE_PATH}/runtime/include>>
)

target_link_libraries(fastergelu_direct_kernel_op PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:tikicpulib::${SOC_VERSION}>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:c_sec>>
    ascendc_kernels_${RUN_MODE}
    tiling_api
    register
    platform
    ascendalog
    dl
    graph_base
)

install(TARGETS fastergelu_direct_kernel_op
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)