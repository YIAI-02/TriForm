# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(UPER_CHARS A B C D E F G H I J K L M N O P Q R S T U V W X Y Z)
function(string_to_snake str_in snake_out)
  set(str_cam ${str_in})
  foreach(uper_char ${UPER_CHARS})
    string(TOLOWER "${uper_char}" lower_char)
    string(REPLACE ${uper_char} "_${lower_char}" str_cam ${str_cam})
  endforeach()
  string(SUBSTRING ${str_cam} 1 -1 str_cam)
  set(${snake_out} "${str_cam}" PARENT_SCOPE)
endfunction()

function(add_cpu_target)
  cmake_parse_arguments(CPU_TEST "" "OP" "SRC" ${ARGN})
  string_to_snake("${CPU_TEST_OP}" op_snake)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${op_snake}_tiling.h
                     COMMAND python3 ${CMAKE_SOURCE_DIR}/cmake/util/tiling_data_def_build.py
                             ${CMAKE_SOURCE_DIR}/op_host/${op_snake}_tiling.h
                             ${CMAKE_CURRENT_SOURCE_DIR}/${op_snake}_tiling.h
                     DEPENDS ${CMAKE_SOURCE_DIR}/op_host/${op_snake}_tiling.h
  )
  add_custom_target(gen_${op_snake}_tiling_header
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${op_snake}_tiling.h
  )

  add_executable(${op_snake}_cpu ${CPU_TEST_SRC})
  add_dependencies(${op_snake}_cpu gen_${op_snake}_tiling_header)
  target_compile_options(${op_snake}_cpu PRIVATE -g -include ${CMAKE_CURRENT_SOURCE_DIR}/${op_snake}_tiling.h)
  target_link_libraries(${op_snake}_cpu PRIVATE tikicpulib::ascend910B1)
  set_target_properties(${op_snake}_cpu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endfunction()

function(add_npu_target)
  cmake_parse_arguments(NPU_TEST "" "OP" "SRC" ${ARGN})
  string_to_snake("${NPU_TEST_OP}" op_snake)
  add_executable(${op_snake}_npu ${NPU_TEST_SRC})
  target_compile_options(${op_snake}_npu PRIVATE -g)
  target_include_directories(${op_snake}_npu PRIVATE
      ${ASCEND_CANN_PACKAGE_PATH}/include/acl
      ${ASCEND_AUTOGEN_PATH}
  )
  target_link_libraries(${op_snake}_npu PRIVATE
      intf_pub
      cust_opapi
      ascendcl
      nnopbase
  )
  set_target_properties(${op_snake}_npu PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
  )
endfunction()
