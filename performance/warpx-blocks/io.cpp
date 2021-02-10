#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <fstream>
#include <iostream>
#include <list>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "io.h"

#include "adios2.h"

IO::IO(const WarpxSettings &settings, const Decomp &decomp, MPI_Comm comm,
       const bool isWriter)
: settings(settings), decomp(decomp), comm(comm), isWriter(isWriter)
{
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    CalculateMyBlocks();
    AllocateBlocks();
}

void IO::CalculateMyBlocks()
{
    /* pre-calculate how many blocks we have to write/read to manage all
       MPI requests at once */

    int r;
    for (int b = 0; b < decomp.nblocks3D; ++b)
    {
        r = (isWriter ? decomp.blocks3D[b].writerRank
                      : decomp.blocks3D[b].readerRank);
        if (r == rank)
        {
            ++nMyBlocks3D;
        }
    }
    for (int b = 0; b < decomp.nblocks1D; ++b)
    {
        r = (isWriter ? decomp.blocks1D[b].writerRank
                      : decomp.blocks1D[b].readerRank);
        if (r == rank)
        {
            ++nMyBlocks1D;
        }
    }
}

void IO::AllocateBlocks()
{
    int r;
    size_t mybid = 0;

    /* 1D blocks */
    beid.resize(nMyBlocks1D);
    bemx.resize(nMyBlocks1D);
    bemy.resize(nMyBlocks1D);
    bemz.resize(nMyBlocks1D);
    bepx.resize(nMyBlocks1D);
    bepy.resize(nMyBlocks1D);
    bepz.resize(nMyBlocks1D);
    bew.resize(nMyBlocks1D);

    mybid = 0;
    for (int b = 0; b < decomp.nblocks1D; ++b)
    {
        r = (isWriter ? decomp.blocks1D[b].writerRank
                      : decomp.blocks1D[b].readerRank);
        if (r == rank)
        {
            const auto &count = decomp.blocks1D[b].count;
            beid[mybid].resize(count);
            bemx[mybid].resize(count);
            bemy[mybid].resize(count);
            bemz[mybid].resize(count);
            bepx[mybid].resize(count);
            bepy[mybid].resize(count);
            bepz[mybid].resize(count);
            bew[mybid].resize(count);
            ++mybid;
        }
    }

    /* 3D blocks */
    bBx.resize(nMyBlocks3D);
    bBy.resize(nMyBlocks3D);
    bBz.resize(nMyBlocks3D);
    bEx.resize(nMyBlocks3D);
    bEy.resize(nMyBlocks3D);
    bEz.resize(nMyBlocks3D);
    bjx.resize(nMyBlocks3D);
    bjy.resize(nMyBlocks3D);
    bjz.resize(nMyBlocks3D);
    brho.resize(nMyBlocks3D);

    mybid = 0;
    for (int b = 0; b < decomp.nblocks3D; ++b)
    {
        r = (isWriter ? decomp.blocks3D[b].writerRank
                      : decomp.blocks3D[b].readerRank);
        if (r == rank)
        {
            const auto &block = decomp.blocks3D[b];
            size_t count = block.count[0] * block.count[1] * block.count[2];
            bBx[mybid].resize(count);
            bBy[mybid].resize(count);
            bBz[mybid].resize(count);
            bEx[mybid].resize(count);
            bEy[mybid].resize(count);
            bEz[mybid].resize(count);
            bjx[mybid].resize(count);
            bjy[mybid].resize(count);
            bjz[mybid].resize(count);
            brho[mybid].resize(count);
            ++mybid;
        }
    }
}
/* Write side "compute function", separated for the purpose of
   measuring computeTime separately */

void IO::Compute(const int step)
{
    int mybid = 0;

    const auto tTotalStart = std::chrono::steady_clock::now();

    /* 3D variables */
    for (int b = 0; b < decomp.nblocks3D; ++b)
    {
        const auto &block = decomp.blocks3D[b];
        if (block.writerRank != rank)
        {
            continue;
        }

        size_t blockSize = block.count[0] * block.count[1] * block.count[2];
        double value = rank + step / 100.0;
        for (size_t i = 0; i < blockSize; ++i)
        {
            bBx[mybid][i] = value;
            bBy[mybid][i] = value;
            bBz[mybid][i] = value;
            bEx[mybid][i] = value;
            bEy[mybid][i] = value;
            bEz[mybid][i] = value;
            bjx[mybid][i] = value;
            bjy[mybid][i] = value;
            bjz[mybid][i] = value;
            brho[mybid][i] = value;
        }
        ++mybid;
    }

    /* 1D variables */
    mybid = 0;
    for (int b = 0; b < decomp.nblocks1D; ++b)
    {
        const auto &block = decomp.blocks1D[b];
        if (block.writerRank != rank)
        {
            continue;
        }

        size_t blockSize = block.count;
        double value = rank + step / 100.0;
        for (size_t i = 0; i < blockSize; ++i)
        {
            beid[mybid][i] = value;
            bemx[mybid][i] = value;
            bemy[mybid][i] = value;
            bemz[mybid][i] = value;
            bepx[mybid][i] = value;
            bepy[mybid][i] = value;
            bepz[mybid][i] = value;
            bew[mybid][i] = value;
        }
        ++mybid;
    }

    const auto tTotalEnd = std::chrono::steady_clock::now();
    Seconds timeTotal = tTotalEnd - tTotalStart;
    Seconds timeToIdle = settings.computeTime - timeTotal;
    if (timeToIdle.count() > 0.0)
    {
        std::this_thread::sleep_for(timeToIdle);
    }
}
