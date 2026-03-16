#ifndef FORMAT_CONV_BENCH_KERNEL_H
#define FORMAT_CONV_BENCH_KERNEL_H

#include "kernel_operator.h"

constexpr uint32_t CUBE_BLOCK = 16;
constexpr uint32_t CUBE_TILE_ELEMS = CUBE_BLOCK * CUBE_BLOCK;

enum FormatConvMode : uint32_t {
    MODE_ND2NZ_A  = 0,
    MODE_ND2NZ_B  = 1,
    MODE_NZ2ZZ_A  = 2,
    MODE_NZ2ZN_B  = 3,
    MODE_ND2ZZ_A  = 4,
    MODE_ND2ZN_B  = 5,
};

class KernelFormatConvBench {
public:
    __aicore__ inline KernelFormatConvBench() {}

    __aicore__ inline void SetShape(uint32_t rowsIn, uint32_t colsIn, uint32_t modeIn, uint32_t innerLoopsIn)
    {
        rows = rowsIn;
        cols = colsIn;
        mode = modeIn;
        innerLoops = (innerLoopsIn == 0 ? 1 : innerLoopsIn);
        rowBlocks = CeilDiv(rows, CUBE_BLOCK);
        colBlocks = CeilDiv(cols, CUBE_BLOCK);
        tiles = rowBlocks * colBlocks;
    }

    __aicore__ inline void Init(GM_ADDR src, GM_ADDR scratch)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
        srcPtr = (__gm__ half*)src;
        scratchPtr = (__gm__ half*)scratch;

        pipe.InitBuffer(qA1, 1, CUBE_TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(qA2, 1, CUBE_TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(qB1, 1, CUBE_TILE_ELEMS * sizeof(half));
        pipe.InitBuffer(qB2, 1, CUBE_TILE_ELEMS * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        switch (mode) {
            case MODE_ND2NZ_A:
                BenchNd2NzA();
                break;
            case MODE_ND2NZ_B:
                BenchNd2NzB();
                break;
            case MODE_NZ2ZZ_A:
                BenchNz2ZzA();
                break;
            case MODE_NZ2ZN_B:
                BenchNz2ZnB();
                break;
            case MODE_ND2ZZ_A:
                BenchNd2ZzA();
                break;
            case MODE_ND2ZN_B:
                BenchNd2ZnB();
                break;
            default:
                BenchNd2ZnB();
                break;
        }
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y) const
    {
        return (x + y - 1) / y;
    }

    __aicore__ inline uint32_t TileSizeAt(uint32_t base, uint32_t full) const
    {
        return ((base + CUBE_BLOCK) <= full) ? CUBE_BLOCK : (full - base);
    }

    __aicore__ inline uint32_t NzTileOffsetElems(uint32_t rb, uint32_t cb) const
    {
        return (cb * rowBlocks + rb) * CUBE_TILE_ELEMS;
    }

    __aicore__ inline void MakeTileTensor(AscendC::GlobalTensor<half>& gt, uint32_t offsetElems, uint32_t lenElems)
    {
        gt.SetGlobalBuffer(srcPtr + offsetElems, lenElems);
    }

    __aicore__ inline void BenchNd2NzA()
    {
        AscendC::LocalTensor<half> a1Local = qA1.AllocTensor<half>();
        AscendC::Nd2NzParams p{};
        p.ndNum = 1;
        p.srcNdMatrixStride = 0;
        p.dstNzC0Stride = CUBE_BLOCK;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;

        for (uint32_t it = 0; it < innerLoops; ++it) {
            for (uint32_t cb = 0; cb < colBlocks; ++cb) {
                const uint32_t c0 = cb * CUBE_BLOCK;
                const uint32_t colsTile = TileSizeAt(c0, cols);
                for (uint32_t rb = 0; rb < rowBlocks; ++rb) {
                    const uint32_t r0 = rb * CUBE_BLOCK;
                    const uint32_t rowsTile = TileSizeAt(r0, rows);
                    AscendC::GlobalTensor<half> ndTile;
                    MakeTileTensor(ndTile, r0 * cols + c0, (rowsTile - 1) * cols + colsTile);
                    p.nValue = rowsTile;
                    p.dValue = colsTile;
                    p.srcDValue = cols;
                    AscendC::DataCopy(a1Local, ndTile, p);
                }
            }
        }
        qA1.FreeTensor(a1Local);
    }

    __aicore__ inline void BenchNd2NzB()
    {
        AscendC::LocalTensor<half> b1Local = qB1.AllocTensor<half>();
        AscendC::Nd2NzParams p{};
        p.ndNum = 1;
        p.srcNdMatrixStride = 0;
        p.dstNzC0Stride = CUBE_BLOCK;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;

        for (uint32_t it = 0; it < innerLoops; ++it) {
            for (uint32_t cb = 0; cb < colBlocks; ++cb) {
                const uint32_t c0 = cb * CUBE_BLOCK;
                const uint32_t colsTile = TileSizeAt(c0, cols);
                for (uint32_t rb = 0; rb < rowBlocks; ++rb) {
                    const uint32_t r0 = rb * CUBE_BLOCK;
                    const uint32_t rowsTile = TileSizeAt(r0, rows);
                    AscendC::GlobalTensor<half> ndTile;
                    MakeTileTensor(ndTile, r0 * cols + c0, (rowsTile - 1) * cols + colsTile);
                    p.nValue = rowsTile;
                    p.dValue = colsTile;
                    p.srcDValue = cols;
                    AscendC::DataCopy(b1Local, ndTile, p);
                }
            }
        }
        qB1.FreeTensor(b1Local);
    }

    __aicore__ inline void BenchNz2ZzA()
    {
        AscendC::LocalTensor<half> a1Local = qA1.AllocTensor<half>();
        AscendC::LocalTensor<half> a2Local = qA2.AllocTensor<half>();
        AscendC::LoadData2DParams p{};
        p.repeatTimes = 1;
        p.srcStride = 1;
        p.dstGap = 0;
        p.ifTranspose = false;

        for (uint32_t it = 0; it < innerLoops; ++it) {
            for (uint32_t cb = 0; cb < colBlocks; ++cb) {
                for (uint32_t rb = 0; rb < rowBlocks; ++rb) {
                    AscendC::GlobalTensor<half> nzTile;
                    MakeTileTensor(nzTile, NzTileOffsetElems(rb, cb), CUBE_TILE_ELEMS);
                    AscendC::DataCopy(a1Local, nzTile, CUBE_TILE_ELEMS);
                    AscendC::LoadData(a2Local, a1Local, p);
                }
            }
        }
        qA1.FreeTensor(a1Local);
        qA2.FreeTensor(a2Local);
    }

    __aicore__ inline void BenchNz2ZnB()
    {
        AscendC::LocalTensor<half> b1Local = qB1.AllocTensor<half>();
        AscendC::LocalTensor<half> b2Local = qB2.AllocTensor<half>();
        AscendC::LoadData2DParams p{};
        p.repeatTimes = 1;
        p.srcStride = 1;
        p.dstGap = 0;
        p.ifTranspose = true;

        for (uint32_t it = 0; it < innerLoops; ++it) {
            for (uint32_t cb = 0; cb < colBlocks; ++cb) {
                for (uint32_t rb = 0; rb < rowBlocks; ++rb) {
                    AscendC::GlobalTensor<half> nzTile;
                    MakeTileTensor(nzTile, NzTileOffsetElems(rb, cb), CUBE_TILE_ELEMS);
                    AscendC::DataCopy(b1Local, nzTile, CUBE_TILE_ELEMS);
                    AscendC::LoadData(b2Local, b1Local, p);
                }
            }
        }
        qB1.FreeTensor(b1Local);
        qB2.FreeTensor(b2Local);
    }

    __aicore__ inline void BenchNd2ZzA()
    {
        AscendC::LocalTensor<half> a1Local = qA1.AllocTensor<half>();
        AscendC::LocalTensor<half> a2Local = qA2.AllocTensor<half>();
        AscendC::Nd2NzParams copyP{};
        copyP.ndNum = 1;
        copyP.srcNdMatrixStride = 0;
        copyP.dstNzC0Stride = CUBE_BLOCK;
        copyP.dstNzNStride = 1;
        copyP.dstNzMatrixStride = 0;
        AscendC::LoadData2DParams loadP{};
        loadP.repeatTimes = 1;
        loadP.srcStride = 1;
        loadP.dstGap = 0;
        loadP.ifTranspose = false;

        for (uint32_t it = 0; it < innerLoops; ++it) {
            for (uint32_t cb = 0; cb < colBlocks; ++cb) {
                const uint32_t c0 = cb * CUBE_BLOCK;
                const uint32_t colsTile = TileSizeAt(c0, cols);
                for (uint32_t rb = 0; rb < rowBlocks; ++rb) {
                    const uint32_t r0 = rb * CUBE_BLOCK;
                    const uint32_t rowsTile = TileSizeAt(r0, rows);
                    AscendC::GlobalTensor<half> ndTile;
                    MakeTileTensor(ndTile, r0 * cols + c0, (rowsTile - 1) * cols + colsTile);
                    copyP.nValue = rowsTile;
                    copyP.dValue = colsTile;
                    copyP.srcDValue = cols;
                    AscendC::DataCopy(a1Local, ndTile, copyP);
                    AscendC::LoadData(a2Local, a1Local, loadP);
                }
            }
        }
        qA1.FreeTensor(a1Local);
        qA2.FreeTensor(a2Local);
    }

    __aicore__ inline void BenchNd2ZnB()
    {
        AscendC::LocalTensor<half> b1Local = qB1.AllocTensor<half>();
        AscendC::LocalTensor<half> b2Local = qB2.AllocTensor<half>();
        AscendC::Nd2NzParams copyP{};
        copyP.ndNum = 1;
        copyP.srcNdMatrixStride = 0;
        copyP.dstNzC0Stride = CUBE_BLOCK;
        copyP.dstNzNStride = 1;
        copyP.dstNzMatrixStride = 0;
        AscendC::LoadData2DParams loadP{};
        loadP.repeatTimes = 1;
        loadP.srcStride = 1;
        loadP.dstGap = 0;
        loadP.ifTranspose = true;

        for (uint32_t it = 0; it < innerLoops; ++it) {
            for (uint32_t cb = 0; cb < colBlocks; ++cb) {
                const uint32_t c0 = cb * CUBE_BLOCK;
                const uint32_t colsTile = TileSizeAt(c0, cols);
                for (uint32_t rb = 0; rb < rowBlocks; ++rb) {
                    const uint32_t r0 = rb * CUBE_BLOCK;
                    const uint32_t rowsTile = TileSizeAt(r0, rows);
                    AscendC::GlobalTensor<half> ndTile;
                    MakeTileTensor(ndTile, r0 * cols + c0, (rowsTile - 1) * cols + colsTile);
                    copyP.nValue = rowsTile;
                    copyP.dValue = colsTile;
                    copyP.srcDValue = cols;
                    AscendC::DataCopy(b1Local, ndTile, copyP);
                    AscendC::LoadData(b2Local, b1Local, loadP);
                }
            }
        }
        qB1.FreeTensor(b1Local);
        qB2.FreeTensor(b2Local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::A1, 1> qA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> qA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> qB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> qB2;

    __gm__ half* srcPtr = nullptr;
    __gm__ half* scratchPtr = nullptr;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t mode = 0;
    uint32_t innerLoops = 1;
    uint32_t rowBlocks = 0;
    uint32_t colBlocks = 0;
    uint32_t tiles = 0;
};

#endif
