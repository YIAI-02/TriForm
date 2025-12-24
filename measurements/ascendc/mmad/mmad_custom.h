/**
 * @file mmad_custom.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef MMAD_CUSTOM_H
#define MMAD_CUSTOM_H
#include "kernel_operator.h"

class KernelMmad {
public:
    __aicore__ inline KernelMmad() //77,106,0.040000
    {
        aSize = m * k;
        bSize = k * n;
        biasSize = n;
        cSize = m * n;
        mBlocks = m / 16;
        nBlocks = n / 16;
        kBlocks = k / 16;
    }
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c)
    {
        aGM.SetGlobalBuffer((__gm__ half*)a);
        bGM.SetGlobalBuffer((__gm__ half*)b);
        biasGM.SetGlobalBuffer((__gm__ half*)bias);
        cGM.SetGlobalBuffer((__gm__ float*)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half) / 2);
        pipe.InitBuffer(outQueueCO1, 2, cSize * sizeof(float) / 2);
        pipe.InitBuffer(outQueueCO2, 1, cSize * sizeof(float));
        pipe.InitBuffer(inQueueBias, 1, biasSize * sizeof(half));
        pipe.InitBuffer(bufBias, biasSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn(); //177,2724,0.050000
        SplitA(); //102,109,0.040000

        AscendC::LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        AscendC::LocalTensor<half> biasLocal = inQueueBias.DeQue<half>();
        AscendC::LocalTensor<float> c2Local = outQueueCO2.AllocTensor<float>();
        // split matrix b into 2 parts, [32, 16] and [32, 16]
        for (int i = 0; i < 2; ++i) {
            SplitB(b1Local, i); //96,118,0.030000
            Compute(a2Local, biasLocal[i * biasSize / 2]);//246,318,0.130000
            Aggregate(c2Local, i); //83,102,0.030000
        }
        inQueueB1.FreeTensor(b1Local);
        inQueueA2.FreeTensor(a2Local);
        inQueueBias.FreeTensor(biasLocal);
        outQueueCO2.EnQue<float>(c2Local);

        CopyOut(); //60,743,0.020000
    }

private:
    __aicore__ inline void CopyND2NZ(const AscendC::LocalTensor<half>& dst, const AscendC::GlobalTensor<half>& src,
        const uint16_t height, const uint16_t width)
    {
        for (int i = 0; i < width / 16; ++i) {
            int srcOffset = i * 16;
            int dstOffset = i * 16 * height;
            AscendC::DataCopy(dst[dstOffset], src[srcOffset], { height, 1, uint16_t(width / 16 - 1), 0 }); //34,1996,0.000000
        }
    }
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        AscendC::LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
        AscendC::LocalTensor<half> biasLocal = inQueueBias.AllocTensor<half>();

        CopyND2NZ(a1Local, aGM, m, k);//33,909,0.000000
        CopyND2NZ(b1Local, bGM, k, n); //31,1096,0.000000
        AscendC::DataCopy(biasLocal, biasGM, biasSize); //7,628,0.000000

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
        inQueueBias.EnQue(biasLocal);
    }
    __aicore__ inline void SplitA()
    {
        int srcOffset = 0;
        int dstOffset = 0;
        AscendC::LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        AscendC::LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        // transform nz to zz
        for (int i = 0; i < mBlocks; ++i) {
            AscendC::LoadData2DParams loadDataParams;
            loadDataParams.repeatTimes = kBlocks;
            loadDataParams.srcStride = mBlocks;
            loadDataParams.ifTranspose = false;

            AscendC::LoadData(a2Local[dstOffset], a1Local[srcOffset], loadDataParams);

            srcOffset += 16 * 16;
            dstOffset += k * 16;
        }

        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB(const AscendC::LocalTensor<half>& b1Local, const int bSplitIdx)
    {
        AscendC::LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        // transform nz to zn
        AscendC::LoadData2DParams loadDataParams;
        loadDataParams.repeatTimes = kBlocks;
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = true;

        AscendC::LoadData(b2Local, b1Local[bSplitIdx * bSize / 2], loadDataParams);

        inQueueB2.EnQue<half>(b2Local);
    }
    __aicore__ inline void Compute(const AscendC::LocalTensor<half>& a2Local,
        const AscendC::LocalTensor<half>& biasLocal)
    {
        AscendC::LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        AscendC::LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        AscendC::LocalTensor<float> biasCastLocal = bufBias.template Get<float>();
        Cast(biasCastLocal, biasLocal, AscendC::RoundMode::CAST_NONE, biasSize / 2);
        BroadCastVecToMM(c1Local, biasCastLocal, 1, 1, 0, 0);
        BroadCastVecToMM(c1Local[16 * 16], biasCastLocal, 1, 1, 0, 0);
        event_t eventIDVToM = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_M));
        AscendC::SetFlag<AscendC::HardEvent::V_M>(eventIDVToM);
        AscendC::WaitFlag<AscendC::HardEvent::V_M>(eventIDVToM);

        AscendC::MmadParams mmadParams;
        mmadParams.m = m;
        mmadParams.n = n / 2;
        mmadParams.k = k;
        mmadParams.cmatrixInitVal = false; //c 不预清零，bias 初始化进c
        AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams); //api

        outQueueCO1.EnQue<float>(c1Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void Aggregate(const AscendC::LocalTensor<float>& c2Local, const int bSplitIdx)
    {
        AscendC::LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 2;
        AscendC::DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        AscendC::DataCopy(c2Local[bSplitIdx * cSize / 2], c1Local, dataCopyParams, enhancedParams);

        outQueueCO1.FreeTensor(c1Local);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> c2Local = outQueueCO2.DeQue<float>();

        // transform nz to nd
        for (int i = 0; i < nBlocks; ++i) {
            AscendC::DataCopy(cGM[i * 16], c2Local[i * m * 16], { m, 2, 0, uint16_t((nBlocks - 1) * 2) }); //19,714,0.000000
        }

        outQueueCO2.FreeTensor(c2Local);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 2> inQueueB2;
    // dst queue
    AscendC::TQue<AscendC::TPosition::CO1, 2> outQueueCO1;
    AscendC::TQue<AscendC::TPosition::CO2, 1> outQueueCO2;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBias;
    AscendC::TBuf<> bufBias;

    AscendC::GlobalTensor<half> aGM, bGM, biasGM;
    AscendC::GlobalTensor<float> cGM;

    uint16_t m = 32;
    uint16_t n = 32;
    uint16_t k = 32;

    uint16_t aSize, bSize, biasSize, cSize, mBlocks, nBlocks, kBlocks;
};

#endif // MMAD_CUSTOM_H
