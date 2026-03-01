cmake_minimum_required(VERSION 3.16)
project(op_bench_acl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ascend 安装路径（脚本会传入，或自动回退到 /usr/local/Ascend/ascend-toolkit/latest）
if (NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if (DEFINED ENV{ASCEND_TOOLKIT_HOME})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_TOOLKIT_HOME})
    elseif(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    endif()
endif()
message(STATUS "ASCEND_CANN_PACKAGE_PATH = ${ASCEND_CANN_PACKAGE_PATH}")

include_directories(${ASCEND_CANN_PACKAGE_PATH}/include)
link_directories(${ASCEND_CANN_PACKAGE_PATH}/lib64 ${ASCEND_CANN_PACKAGE_PATH}/lib)

add_executable(op_runner main.cpp)

# 1) AscendCL：必须
find_library(ASCENDCL_LIB
    NAMES ascendcl
    HINTS ${ASCEND_CANN_PACKAGE_PATH}/lib64 ${ASCEND_CANN_PACKAGE_PATH}/lib
    NO_DEFAULT_PATH
)
if(NOT ASCENDCL_LIB)
    message(FATAL_ERROR "Failed to find ascendcl under ${ASCEND_CANN_PACKAGE_PATH}")
endif()

# 2) 单算子在线编译库：可选（8.x 通常存在）
find_library(ACLOP_COMPILER_LIB
    NAMES acl_op_compiler
    HINTS ${ASCEND_CANN_PACKAGE_PATH}/lib64 ${ASCEND_CANN_PACKAGE_PATH}/lib
    NO_DEFAULT_PATH
)

# 链接顺序：对象 -> ascendcl ->（可选）acl_op_compiler -> pthread dl
target_link_libraries(op_runner PRIVATE ${ASCENDCL_LIB} pthread dl)
if (ACLOP_COMPILER_LIB)
    target_link_libraries(op_runner PRIVATE ${ACLOP_COMPILER_LIB})
    message(STATUS "Found libacl_op_compiler: ${ACLOP_COMPILER_LIB}")
endif()

# 让可执行文件在运行时能找到库
set_target_properties(op_runner PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib64")

install(TARGETS op_runner RUNTIME DESTINATION bin)
