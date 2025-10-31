#!/bin/bash
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

script_path=$(realpath $(dirname $0))

source $ASCEND_HOME_DIR/bin/setenv.bash
cp -rf ../host_tiling/* op_host/
rm -rf ./cmake/util
ln -s $ASCEND_HOME_DIR/tools/op_project_templates/ascendc/customize/cmake/util/ ./cmake/util
mkdir -p build_out
rm -rf build_out/*
cd build_out

opts=$(python3 $script_path/cmake/util/preset_parse.py $script_path/CMakePresets.json)
ENABLE_CROSS="-DENABLE_CROSS_COMPILE=True"
ENABLE_BINARY="-DENABLE_BINARY_PACKAGE=True"
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

cmake_run_package()
{
  target=$1
  cmake --build . --target $target -j16
  if [ $? -ne 0 ]; then exit 1; fi

  if [ $target = "package" ]; then
    if test -d ./op_kernel/binary ; then
      ./cust*.run
      if [ $? -ne 0 ]; then exit 1; fi
      cmake --build . --target binary -j16
      if [ $? -ne 0 ]; then exit 1; fi
      cmake --build . --target $target -j16
    fi
  fi
}

if [[ $opts =~ $ENABLE_CROSS ]] && [[ $opts =~ $ENABLE_BINARY ]]
then
  target=package
  if [ "$1"x != ""x ]; then target=$1; fi
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake .. $opts -DENABLE_CROSS_COMPILE=0
  else
    cmake .. --preset=default -DENABLE_CROSS_COMPILE=0
  fi
  cmake_run_package $target
  cp -r kernel ../
  rm -rf *
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake .. $opts
  else
    cmake .. --preset=default
  fi

  cmake --build . --target $target -j16
  if [ $? -ne 0 ]; then exit 1; fi
  if [ $target = "package" ]; then
    if test -d ./op_kernel/binary ; then
      ./cust*.run
    fi
  fi
  rm -rf ../kernel

else
  target=package
  if [ "$1"x != ""x ]; then target=$1; fi
  if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake .. $opts
  else
      cmake .. --preset=default
  fi
  cmake_run_package $target
fi


# for debug
# cd build_out
# make
# cpack
# verbose append -v