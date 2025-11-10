<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->

## 概述

本样例基于Kernel直调算子工程，介绍了调用LayerNorm高阶API实现layernorm单算子，主要演示LayerNorm高阶API在Kernel直调工程中的调用。

## 目录结构介绍
| 目录及文件                  | 描述                   |
|---------------------|----------------------|
| [cmake](./cmake)      | 编译工程文件 |
| [scripts](./scripts) | 包含输入数据和真值数据生成脚本文件 |
| main.cpp | 主函数，调用算子的应用程序，含CPU域及NPU域调用 |
| layernorm_custom.cpp | 算子kernel实现 |
| layernorm_custom_tiling.cpp | 算子tiling实现 |
| run.sh | 编译执行脚本 |
| CMakeLists.txt | 编译工程文件 |


## 编译运行样例

  - 配置环境变量

    这里的\$ASCEND_CANN_PACKAGE_PATH需要替换为CANN开发套件包安装后文件存储路径。例如：/usr/local/Ascend/ascend-toolkit/latest
    ```
    export ASCEND_HOME_DIR=$ASCEND_CANN_PACKAGE_PATH
    source $ASCEND_HOME_DIR/../set_env.sh
    ```

  - 生成输入和真值

    执行如下命令后，当前目录生成input和output目录存放输入数据和真值数据。
    ```
    python3 scripts/gen_data.py
    ```

  - 编译执行

    ```
    bash run.sh -r [RUN_MODE] -v [SOC_VERSION]
    ```
    其中脚本参数说明如下：
    - RUN_MODE ：编译执行方式，可选择CPU调试，NPU仿真，NPU上板，对应参数分别为[cpu / sim/ npu]。若需要详细了解NPU仿真相关内容，请参考[《算子开发工具msProf》](https://hiascend.com/document/redirect/CannCommunityToolMsProf)中的“工具使用”章节。
    - SOC_VERSION ：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas 推理系列产品AI Core
      - Atlas A2训练系列产品/Atlas 800I A2推理产品

    示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
    ```
    bash run.sh -r cpu -v Ascendxxxyy
    ```