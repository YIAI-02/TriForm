<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->

## 概述

本样例基于自定义算子工程，介绍了调用LayerNorm高阶API实现layernorm单算子，主要演示LayerNorm高阶API在自定义算子工程中的调用。

## 目录结构
| 目录                  | 描述                   |
|---------------------|----------------------|
| [cmake](./cmake)      | 编译工程文件 |
| [op_host](./op_host)       | host侧实现文件 |
| [op_kernel](./op_kernel) | kernel侧实现文件 |
| [scripts](./scripts) | 包含输入数据和真值数据生成脚本文件 |
| [testcases](./testcases) | 包含cpu域以及npu域的用例主函数，以及真值校验函数 |
| build.sh | 编译运行算子的脚本 |
| CMakeLists.txt | 编译工程文件 |
| CMakePresets.json | 编译工程配置文件 |

## 编译运行样例

### 1.配置环境变量

  这里的\$ASCEND_CANN_PACKAGE_PATH需要替换为CANN开发套件包安装后文件存储路径。例如：/usr/local/Ascend/ascend-toolkit/latest
  ```
  export ASCEND_HOME_DIR=$ASCEND_CANN_PACKAGE_PATH
  source $ASCEND_HOME_DIR/../set_env.sh
  ```
### 2.生成输入和真值
  执行如下命令后，当前目录生成input和output目录存放输入数据和真值数据。
  ```
  python3 scripts/gen_data.py
  ```

### 3.编译算子工程

  - 修改CMakePresets.json中ASCEND_CANN_PACKAGE_PATH为CANN开发套件包安装后文件存储路径。


    ```
    {
        ……
        "configurePresets": [
            {
                    ……
                    "ASCEND_CANN_PACKAGE_PATH": {
                        "type": "PATH",
                        //请替换为CANN开发套件包安装后文件存储路径。例如：/usr/local/Ascend/ascend-toolkit/latest
                        "value": "~/Ascend/ascend-toolkit/latest"
                    },
                    ……
            }
        ]
    }
    ```
  - 在当前算子工程目录下执行如下命令，进行算子工程编译。

    ```
    bash build.sh
    ```
    编译成功后，会在当前目录下创建build_out目录，并在build_out目录下生成自定义算子安装包custom_opp_\<target os>_\<target architecture>.run，例如“custom_opp_ubuntu_x86_64.run”。


### 4.部署算子包

  - 执行如下命令，在自定义算子安装包所在路径下，安装自定义算子包。

    ```
    cd build_out
    ./custom_opp_<target os>_<target architecture>.run
    ```

    命令执行成功后，自定义算子包中的相关文件将部署至当前环境的OPP算子库的vendors/customize目录中。
### 5.执行样例
  - 在build_out目录下执行如下命令

    ```
    ./layernorm_v2_custom_npu
    ```

### 6.NPU仿真模式运行（可选）
若要执行NPU仿真，在build_out目录下执行如下命令：
```
export LD_LIBRARY_PATH=$ASCEND_HOME_DIR/tools/simulator/${SOC_VERSION}/lib:$LD_LIBRARY_PATH
msprof op simulator --application=./layernorm_v2_custom_npu
```
其中SOC_VERSION参数说明如下：
- SOC_VERSION ：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
  - Atlas 推理系列产品AI Core
  - Atlas A2训练系列产品/Atlas 800I A2推理产品

若需要详细了解NPU仿真相关内容，请参考[《算子开发工具msProf》](https://hiascend.com/document/redirect/CannCommunityToolMsProf)中的“工具使用”章节。
### 7.不同环境上的编译与运行（可选）
若想在不同的环境上分别进行编译和执行，请在执行环境中进行如下操作，确保该环境上能够正确执行样例.
注意，以下方法仅支持编译环境与运行环境是相同的物理硬件架构，比如编译环境和执行环境均为x86硬件架构；若硬件架构不一致，必须重新编译算子工程，再安装部署和运行样例。
  - 参考步骤1，配置环境变量。
  - 参考步骤2，生成输入和真值数据，或者将编译环境下生成的input和output目录复制到执行环境。
  - 将编译环境下编译生成的自定义算子包和可执行程序，复制到执行环境。
  - 参考步骤4，在执行环境，安装部署自定义算子包。
  - 设置如下环境变量：
    ```
    export LD_LIBRARY_PATH=$ASCEND_HOME_DIR/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
    ```
  - 在input/output的同级目录中创建一个临时目录，将可执行程序放入临时目录，进入临时目录参考步骤5，执行可执行程序，即可运行样例。