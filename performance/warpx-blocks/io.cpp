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
    nMyBlocks3D = 0;
    for (int b = 0; b < decomp.nblocks3D; ++b)
    {
        r = (isWriter ? decomp.blocks3D[b].writerRank
                      : decomp.blocks3D[b].readerRank);
        if (r == rank)
        {
            ++nMyBlocks3D;
        }
    }
    nMyBlocks1D = 0;
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
            nTotalAllocatedSize3D += 8 * count * sizeof(double);
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
            nTotalAllocatedSize1D += 10 * count * sizeof(double);
            ++mybid;
        }
    }
    nTotalAllocatedSize = nTotalAllocatedSize1D + nTotalAllocatedSize3D;
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

/* Copy a block into the larger variable array
   The block is assumed to be completely inside the array */
void IO::Copy3D(std::vector<double> &var3d, const struct ReaderDecomp &vd,
                const std::vector<double> &blockdata,
                const struct Block3D &block, const int blockid)
{
    static int msgs = 0;
    const int msglimit = 100;
    int blockSize =
        static_cast<int>(block.count[0] * block.count[1] * block.count[2]);
    if (settings.verbose >= 2 && msgs < msglimit)
    {
        std::cout << "Reader rank: " << rank << " copy 3D block " << blockid
                  << " blockSize = " << blockSize << " start = {"
                  << block.start[0] << ", " << block.start[1] << ", "
                  << block.start[2] << "}"
                  << " count = {" << block.count[0] << ", " << block.count[1]
                  << ", " << block.count[2] << "}"
                  << " into 3D variable start = {" << vd.start3D[0] << ", "
                  << vd.start3D[1] << ", " << vd.start3D[2] << "}"
                  << " count = {" << vd.count3D[0] << ", " << vd.count3D[1]
                  << ", " << vd.count3D[2] << "}" << std::endl;
    }
    if (var3d.size() != vd.nElems3D)
    {
        throw std::runtime_error(
            "Coding error: passed the wrong variable vector in Copy3D()");
    }

    /* Local start offsets of block in local array (block, vd are global)*/
    size_t start[3], end[3];
    for (int i = 0; i < 3; ++i)
    {
        start[i] = block.start[i] - vd.start3D[i];
        end[i] = start[i] + block.count[i] - 1;
        if (end[i] > vd.count3D[i])
        {
            throw std::runtime_error(
                "Block does not fit into variable, a requirement for Copy3D()");
        }
    }

    bool contZ = false;  /* contiguous in Z dim? */
    bool contYZ = false; /* contiguous in Y and Z dim? */

    if (start[2] == 0 && end[2] == vd.count3D[2] - 1)
    {
        contZ = true;
    }

    if (contZ && start[1] == 0 && end[1] == vd.count3D[1] - 1)
    {
        contYZ = true;
    }

    if (settings.verbose >= 2 && msgs < msglimit)
    {
        std::cout << "Reader rank " << rank << " block local coords"
                  << " start = {" << start[0] << ", " << start[1] << ", "
                  << start[2] << "}"
                  << " end = {" << end[0] << ", " << end[1] << ", " << end[2]
                  << "}"
                  << " varend = {" << vd.count3D[0] - 1 << ", "
                  << vd.count3D[1] - 1 << ", " << vd.count3D[2] - 1 << "}"
                  << std::endl;
    }

    if (contYZ)
    {
        /* Copy the whole block at once */
        if (settings.verbose >= 2 && msgs < msglimit)
        {
            std::cout << "Reader rank " << rank
                      << " block is contiguous in YZ dims" << std::endl;
        }
        size_t offset = start[0] * vd.count3D[1] * vd.count3D[2];
        std::copy(blockdata.begin(), blockdata.end(), var3d.begin() + offset);
    }
    else if (contZ)
    {
        /* Copy 2D planes at once */
        if (settings.verbose >= 2 && msgs < msglimit)
        {
            std::cout << "Reader rank " << rank
                      << " block is contiguous in Z dim" << std::endl;
        }
        for (size_t x = 0; x < block.count[0]; ++x)
        {
            size_t nelems = block.count[1] * block.count[2];
            size_t boffset = x * block.count[1] * block.count[2];
            size_t voffset = (start[0] + x) * vd.count3D[1] * vd.count3D[2] +
                             start[1] * vd.count3D[2];
            std::copy(blockdata.begin() + boffset,
                      blockdata.begin() + boffset + nelems - 1,
                      var3d.begin() + voffset);
        }
    }
    else
    {
        /* Copy each Z-row at once */
        if (settings.verbose >= 2 && msgs < msglimit)
        {
            std::cout << "Reader rank " << rank
                      << " block has to be copied row by row" << std::endl;
        }
        for (size_t x = 0; x < block.count[0]; ++x)
        {
            for (size_t y = 0; y < block.count[1]; ++y)
            {
                size_t nelems = block.count[2];
                size_t boffset =
                    x * block.count[1] * block.count[2] + y * block.count[2];
                size_t voffset =
                    (start[0] + x) * vd.count3D[1] * vd.count3D[2] +
                    (start[1] + y) * vd.count3D[2];
                if (voffset + nelems > var3d.size())
                {
                    if (msgs < msglimit)
                    {
                        std::cout << "  ERROR x = " << x << " y = " << y
                                  << " voffset = " << voffset
                                  << " boffset = " << boffset
                                  << " nelems = " << nelems
                                  << " blocksize = " << var3d.size()
                                  << std::endl;
                        ++msgs;
                    }
                }
                std::copy(blockdata.begin() + boffset,
                          blockdata.begin() + boffset + nelems - 1,
                          var3d.begin() + voffset);
            }
        }
    }
    ++msgs;
}
