
cmake_minimum_required(VERSION 3.16)
project(op_bench_acl CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow user to pass CANN toolkit path in (default: $ASCEND_TOOLKIT_HOME or /usr/local/Ascend/ascend-toolkit/latest)
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

add_executable(op_runner src/main.cpp)

# Link AscendCL (runtime + single-op API)
find_library(ASCENDCL_LIB ascendcl HINTS ${ASCEND_CANN_PACKAGE_PATH}/lib64 ${ASCEND_CANN_PACKAGE_PATH}/lib NO_DEFAULT_PATH)
if(NOT ASCENDCL_LIB)
    message(FATAL_ERROR "Failed to find ascendcl library under ${ASCEND_CANN_PACKAGE_PATH}")
endif()

target_link_libraries(op_runner ${ASCENDCL_LIB} pthread dl)
# RPATH so the binary can find libascendcl.so at runtime
set_target_properties(op_runner PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../lib64")
install(TARGETS op_runner RUNTIME DESTINATION bin)
