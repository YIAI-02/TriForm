/**
 * @file mmad_custom_cube_only.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef MMAD_CUSTOM_CUBE_ONLY_H
#define MMAD_CUSTOM_CUBE_ONLY_H
#include "kernel_operator.h"

// half type, cube block: [16, 16]
constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t CUBE_BLOCK_SIZE = 16 * 16;

class KernelMmad {
public:
    __aicore__ inline KernelMmad()
    {
        aSize = m * k;
        bSize = k * n;
        cSize = m * n;
    }
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c)
    {
        // set cube only
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

        aGM.SetGlobalBuffer((__gm__ half *)a);
        bGM.SetGlobalBuffer((__gm__ half *)b);
        cGM.SetGlobalBuffer((__gm__ float *)c);
        biasGM.SetGlobalBuffer((__gm__ half *)bias);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 1, bSize * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
        pipe.InitBuffer(inQueueC1, 1, n * sizeof(half));
        pipe.InitBuffer(outQueueC2, 1, n * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();
        SplitB();
        SplitBias();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline uint32_t CeilCubeBlock(uint32_t len) {
        return (len + CUBE_BLOCK - 1) / CUBE_BLOCK;
    }

    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        AscendC::LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
        AscendC::LocalTensor<half> bias1Local = inQueueC1.AllocTensor<half>();

        AscendC::Nd2NzParams nd2nzA1Params;
        nd2nzA1Params.ndNum = 1;
        nd2nzA1Params.nValue = m;
        nd2nzA1Params.dValue = k;
        nd2nzA1Params.srcNdMatrixStride = 0;
        nd2nzA1Params.srcDValue = k;
        nd2nzA1Params.dstNzC0Stride = CeilCubeBlock(m) * CUBE_BLOCK;
        nd2nzA1Params.dstNzNStride = 1;
        nd2nzA1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(a1Local, aGM, nd2nzA1Params);

        AscendC::Nd2NzParams nd2nzB1Params;
        nd2nzB1Params.ndNum = 1;
        nd2nzB1Params.nValue = k;
        nd2nzB1Params.dValue = n;
        nd2nzB1Params.srcNdMatrixStride = 0;
        nd2nzB1Params.srcDValue = n;
        nd2nzB1Params.dstNzC0Stride = CeilCubeBlock(k) * CUBE_BLOCK;
        nd2nzB1Params.dstNzNStride = 1;
        nd2nzB1Params.dstNzMatrixStride = 0;
        AscendC::DataCopy(b1Local, bGM, nd2nzB1Params);

        AscendC::DataCopy(bias1Local, biasGM, n);
        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
        inQueueC1.EnQue(bias1Local);
    }

    __aicore__ inline void SplitA()
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        uint32_t dstOffset = CeilCubeBlock(k) * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;
 
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(k);
        loadDataParams.srcStride = CeilCubeBlock(m);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = false;
        for (int i = 0; i < CeilCubeBlock(m); ++i) {
            AscendC::LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParams);
        }

        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB()
    {
        AscendC::LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        uint32_t dstOffset = CeilCubeBlock(n) * CUBE_BLOCK_SIZE;
        uint32_t srcOffset = CUBE_BLOCK_SIZE;
        // Nz -> Zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = CeilCubeBlock(n);
        loadDataParams.srcStride = CeilCubeBlock(k);
        loadDataParams.dstGap = 0;
        loadDataParams.ifTranspose = true;
        for (int i = 0; i < CeilCubeBlock(k); ++i) {
            AscendC::LoadData(b2Local[i * dstOffset], b1Local[i * srcOffset], loadDataParams);
        }

        inQueueB1.FreeTensor(b1Local);
        inQueueB2.EnQue<half>(b2Local);
    }
    __aicore__ inline void SplitBias()
    {
        AscendC::LocalTensor<half> bias1Local = inQueueC1.DeQue<half>();
        AscendC::LocalTensor<float> bias2Local = outQueueC2.AllocTensor<float>();
        AscendC::DataCopy(bias2Local, bias1Local, { 1, (uint16_t)(n * sizeof(half) / 64), 0, 0 });
        outQueueC2.EnQue<float>(bias2Local);
        inQueueC1.FreeTensor(bias1Local);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        AscendC::LocalTensor<float> bias2Local = outQueueC2.DeQue<float>();
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n;
        mmadParams.k = k;
        mmadParams.cmatrixInitVal = false;
        AscendC::Mmad(c1Local, a2Local, b2Local, bias2Local, mmadParams);
        outQueueCO1.EnQue<float>(c1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
        outQueueC2.FreeTensor(bias2Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        AscendC::FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;

        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        AscendC::Fixpipe(cGM, c1Local, fixpipeParams);
        outQueueCO1.FreeTensor(c1Local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;
    AscendC::TQue<AscendC::TPosition::C1, 1> inQueueC1;
    AscendC::TQue<AscendC::TPosition::C2, 1> outQueueC2;

    AscendC::GlobalTensor<half> aGM;
    AscendC::GlobalTensor<half> bGM;
    AscendC::GlobalTensor<float> cGM;
    AscendC::GlobalTensor<half> biasGM;
    uint16_t m = 32, k = 32, n = 32;
    uint16_t aSize, bSize, cSize;
};

#endif // MMAD_CUSTOM_CUBE_ONLY_H
