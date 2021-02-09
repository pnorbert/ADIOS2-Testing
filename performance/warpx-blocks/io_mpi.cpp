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

#include "io_mpi.h"

#include "adios2.h"

IO_MPI::IO_MPI(const WarpxSettings &settings, const Decomp &decomp,
               MPI_Comm comm, const bool isWriter)
: settings(settings), decomp(decomp), comm(comm), isWriter(isWriter)
{
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldNProc);
    ExchangeWorldRanks();
    CalculateMyBlocks();
    AllocateBlocks();
}

/* Global communication */
void IO_MPI::ExchangeWorldRanks()
{
    struct Info
    {
        bool isWriter;
        int rank;
        int worldRank;
    };

    readerWorldRanks.resize(settings.nReaders);
    writerWorldRanks.resize(settings.nWriters);

    Info info;
    info.isWriter = isWriter;
    info.rank = rank;
    info.worldRank = worldRank;

    std::vector<Info> allinfo(worldNProc);

    MPI_Allgather(&info, sizeof(Info), MPI_BYTE, allinfo.data(), sizeof(Info),
                  MPI_BYTE, MPI_COMM_WORLD);

    for (int i = 0; i < worldNProc; ++i)
    {
        if (allinfo[i].isWriter)
        {
            writerWorldRanks[allinfo[i].rank] = allinfo[i].worldRank;
        }
        else
        {
            readerWorldRanks[allinfo[i].rank] = allinfo[i].worldRank;
        }
    }
}

void IO_MPI::CalculateMyBlocks()
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

/* Write side "compute function", separated for the purpose of
   measuring computeTime separately */
void IO_MPI::Compute(const int step)
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
        double value = 10 + rank + step / 100.0;
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

void IO_MPI::AllocateBlocks()
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

void IO_MPI::WriterMPI()
{
    MPI_Request req[10 * nMyBlocks3D + 8 * nMyBlocks1D];
    MPI_Status statuses[10 * nMyBlocks3D + 8 * nMyBlocks1D];
    if (settings.verbose >= 2)
    {
        std::cout << "Writer rank: " << rank << " will have "
                  << 10 * nMyBlocks3D + 8 * nMyBlocks1D << " requests "
                  << std::endl;
    }

    for (int step = 0; step < settings.steps; ++step)
    {
        if (!rank)
        {
            std::cout << "Writer Step: " << step << std::endl;
        }

        Compute(step);

        int reqCount = 0;
        int mybid = 0;

        /* 3D variables */
        for (int b = 0; b < decomp.nblocks3D; ++b)
        {
            const auto &block = decomp.blocks3D[b];
            if (block.writerRank != rank)
            {
                continue;
            }

            size_t blockSize = block.count[0] * block.count[1] * block.count[2];
            int sendTo = readerWorldRanks[block.readerRank];
            int tag = b * 100;
            if (settings.verbose >= 2)
            {
                std::cout << "Writer rank: " << rank << " isend 3D block " << b
                          << " blockSize = " << blockSize << " to reader "
                          << block.readerRank << " (wrank=" << sendTo << ")"
                          << " tag = " << tag << " req = " << reqCount
                          << std::endl;
            }

            MPI_Isend(bBx[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*Bx*/
            MPI_Isend(bBy[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*By*/
            MPI_Isend(bBz[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*Bz*/
            MPI_Isend(bEz[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*Ex*/
            MPI_Isend(bEy[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*Ey*/
            MPI_Isend(bEz[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*Ez*/
            MPI_Isend(bjx[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*jx*/
            MPI_Isend(bjy[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*jy*/
            MPI_Isend(bjz[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*jz*/
            MPI_Isend(brho[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*rho*/
        }

        mybid = 0;

        /* 1D variables */
        for (int b = 0; b < decomp.nblocks1D; ++b)
        {
            const auto &block = decomp.blocks1D[b];
            if (block.writerRank != rank)
            {
                continue;
            }

            size_t blockSize = block.count;
            int sendTo = readerWorldRanks[block.readerRank];
            int tag = b * 100 + 50;
            if (settings.verbose >= 2)
            {
                std::cout << "Writer rank: " << rank << " isend 1D block " << b
                          << " blockSize = " << blockSize << " to reader "
                          << block.readerRank << " (wrank=" << sendTo << ")"
                          << " tag = " << tag << " req = " << reqCount
                          << std::endl;
            }

            MPI_Isend(beid[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*eid*/
            MPI_Isend(bemx[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*emx*/
            MPI_Isend(bemy[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*emy*/
            MPI_Isend(bemz[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*emz*/
            MPI_Isend(bepx[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*epx*/
            MPI_Isend(bepy[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*epy*/
            MPI_Isend(bepz[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*epz*/
            MPI_Isend(bew[mybid].data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*ew*/
            /*if (settings.verbose >= 2)
            {
                std::cout << "Writer rank: " << rank << " Wait for " << reqCount
                          << " requests with tag = " << tag - 8
                          << " reqCount = " << reqCount << std::endl;
            }
            MPI_Waitall(reqCount, req, statuses);
            reqCount = 0;*/

            ++mybid;
        }

        if (settings.verbose >= 2)
        {
            std::cout << "Writer rank: " << rank << " waitall "
                      << " reqCount = " << reqCount << std::endl;
        }
        MPI_Waitall(reqCount, req, statuses);
    }
}

void IO_MPI::ReaderMPI()
{
    MPI_Request req[10 * nMyBlocks3D + 8 * nMyBlocks1D];
    MPI_Status statuses[10 * nMyBlocks3D + 8 * nMyBlocks1D];
    if (settings.verbose >= 2)
    {
        std::cout << "Reader rank: " << rank << " will have "
                  << 10 * nMyBlocks3D + 8 * nMyBlocks1D << " requests "
                  << std::endl;
    }

    std::string inputFileName;
    std::string variableName;
    std::string variableType;

    const ReaderDecomp &d = decomp.readers[rank];
    /* Reader's variables in memory */
    std::vector<double> Bx(d.nElems3D), By(d.nElems3D), Bz(d.nElems3D),
        Ex(d.nElems3D), Ey(d.nElems3D), Ez(d.nElems3D), jx(d.nElems3D),
        jy(d.nElems3D), jz(d.nElems3D), rho(d.nElems3D);

    std::vector<double> eid(d.count1D), emx(d.count1D), emy(d.count1D),
        emz(d.count1D), epx(d.count1D), epy(d.count1D), epz(d.count1D),
        ew(d.count1D);

    adios2::ADIOS adios(settings.adios_config, comm);
    adios2::IO io = adios.DeclareIO("WarpX");
    adios2::Engine dump;
    if (settings.readerDump)
    {
        io.SetEngine("FileStream");
        dump = io.Open("dump_mpi.bp", adios2::Mode::Write);
    }

    adios2::Dims start3d{d.start3D[0], d.start3D[1], d.start3D[2]};
    adios2::Dims count3d{d.count3D[0], d.count3D[1], d.count3D[2]};
    // adios2::Box<adios2::Dims> sel3D = {start3d, count3d};
    adios2::Variable<double> vBx = io.DefineVariable<double>(
        "/data/fields/B/x", decomp.shape3D, start3d, count3d, false);
    adios2::Variable<double> vBy = io.DefineVariable<double>(
        "/data/fields/B/y", decomp.shape3D, start3d, count3d, false);
    adios2::Variable<double> vBz = io.DefineVariable<double>(
        "/data/fields/B/z", decomp.shape3D, start3d, count3d, false);
    adios2::Variable<double> vEx = io.DefineVariable<double>(
        "/data/fields/E/x", decomp.shape3D, start3d, count3d, false);
    adios2::Variable<double> vEy = io.DefineVariable<double>(
        "/data/fields/E/y", decomp.shape3D, start3d, count3d, false);
    adios2::Variable<double> vEz = io.DefineVariable<double>(
        "/data/fields/E/z", decomp.shape3D, start3d, count3d, false);
    adios2::Variable<double> vjx = io.DefineVariable<double>(
        "/data/fields/j/x", decomp.shape3D, start3d, count3d, false);
    adios2::Variable<double> vjy = io.DefineVariable<double>(
        "/data/fields/j/y", decomp.shape3D, start3d, count3d, false);
    adios2::Variable<double> vjz = io.DefineVariable<double>(
        "/data/fields/j/z", decomp.shape3D, start3d, count3d, false);
    adios2::Variable<double> vrho = io.DefineVariable<double>(
        "/data/fields/rho", decomp.shape3D, start3d, count3d, false);

    adios2::Dims shape1d{decomp.shape1D};
    adios2::Dims start1d{d.start1D};
    adios2::Dims count1d{d.count1D};
    // adios2::Box<adios2::Dims> sel1D = { start1d, count1d;
    adios2::Variable<double> veid = io.DefineVariable<double>(
        "/data/800/particles/electrons/id", shape1d, start1d, count1d, false);
    adios2::Variable<double> vemx =
        io.DefineVariable<double>("/data/800/particles/electrons/momentum/x",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vemy =
        io.DefineVariable<double>("/data/800/particles/electrons/momentum/y",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vemz =
        io.DefineVariable<double>("/data/800/particles/electrons/momentum/z",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vepx =
        io.DefineVariable<double>("/data/800/particles/electrons/position/x",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vepy =
        io.DefineVariable<double>("/data/800/particles/electrons/position/y",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vepz =
        io.DefineVariable<double>("/data/800/particles/electrons/position/z",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vew =
        io.DefineVariable<double>("/data/800/particles/electrons/weighting",
                                  shape1d, start1d, count1d, false);

    for (int step = 0; step < settings.steps; ++step)
    {
        if (!rank)
        {
            std::cout << "Reader Step: " << step << std::endl;
        }
        int reqCount = 0;
        size_t mybid = 0;

        /* 3D variables */
        for (int b = 0; b < decomp.nblocks3D; ++b)
        {
            const auto &block = decomp.blocks3D[b];
            if (block.readerRank != rank)
            {
                continue;
            }

            int blockSize = static_cast<int>(block.count[0] * block.count[1] *
                                             block.count[2]);

            int recvFrom = writerWorldRanks[block.writerRank];
            int tag = b * 100;

            if (settings.verbose >= 2)
            {
                std::cout << "Reader rank: " << rank << " irecv 3D block " << b
                          << " blockSize = " << blockSize << " from writer "
                          << block.writerRank << " (wrank=" << recvFrom << ")"
                          << " tag = " << tag << " req = " << reqCount
                          << std::endl;
            }

            MPI_Irecv(bBx[mybid].data(), blockSize, MPI_DOUBLE, recvFrom, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*Bx*/
            MPI_Irecv(bBy[mybid].data(), blockSize, MPI_DOUBLE, recvFrom, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*By*/
            MPI_Irecv(bBz[mybid].data(), blockSize, MPI_DOUBLE, recvFrom, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*Bz*/
            MPI_Irecv(bEx[mybid].data(), blockSize, MPI_DOUBLE, recvFrom, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*Ex*/
            MPI_Irecv(bEy[mybid].data(), blockSize, MPI_DOUBLE, recvFrom, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*Ey*/
            MPI_Irecv(bEz[mybid].data(), blockSize, MPI_DOUBLE, recvFrom, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*Ez*/
            MPI_Irecv(bjx[mybid].data(), blockSize, MPI_DOUBLE, recvFrom, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*jx*/
            MPI_Irecv(bjy[mybid].data(), blockSize, MPI_DOUBLE, recvFrom, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*jy*/
            MPI_Irecv(bjz[mybid].data(), blockSize, MPI_DOUBLE, recvFrom, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*jz*/
            MPI_Irecv(brho[mybid].data(), blockSize, MPI_DOUBLE, recvFrom,
                      tag++, MPI_COMM_WORLD, &req[reqCount++]); /*rho*/
            /*if (settings.verbose >= 2)
            {
                std::cout << "Reader rank: " << rank << " Wait for " << reqCount
                          << " requests with tag = " << tag - 8 << std::endl;
            }
            MPI_Waitall(reqCount, req, statuses);
            reqCount = 0;*/
            ++mybid;
        }

        mybid = 0;
        /* 1D variables */
        for (int b = 0; b < decomp.nblocks1D; ++b)
        {
            const auto &block = decomp.blocks1D[b];
            if (block.readerRank != rank)
            {
                continue;
            }

            int blockSize = static_cast<int>(block.count);

            int recvFrom = writerWorldRanks[block.writerRank];
            int tag = b * 100 + 50;

            if (settings.verbose >= 2)
            {
                std::cout << "Reader rank: " << rank << " irecv 1D block " << b
                          << " blockSize = " << blockSize << " from writer "
                          << block.writerRank << " (wrank=" << recvFrom << ")"
                          << " tag = " << tag << " req = " << reqCount
                          << std::endl;
            }

            MPI_Irecv(beid[mybid].data(), blockSize, MPI_DOUBLE, recvFrom,
                      tag++, MPI_COMM_WORLD, &req[reqCount++]); /*eid*/
            MPI_Irecv(bemx[mybid].data(), blockSize, MPI_DOUBLE, recvFrom,
                      tag++, MPI_COMM_WORLD, &req[reqCount++]); /*emx*/
            MPI_Irecv(bemy[mybid].data(), blockSize, MPI_DOUBLE, recvFrom,
                      tag++, MPI_COMM_WORLD, &req[reqCount++]); /*emy*/
            MPI_Irecv(bemz[mybid].data(), blockSize, MPI_DOUBLE, recvFrom,
                      tag++, MPI_COMM_WORLD, &req[reqCount++]); /*emz*/
            MPI_Irecv(bepx[mybid].data(), blockSize, MPI_DOUBLE, recvFrom,
                      tag++, MPI_COMM_WORLD, &req[reqCount++]); /*epx*/
            MPI_Irecv(bepy[mybid].data(), blockSize, MPI_DOUBLE, recvFrom,
                      tag++, MPI_COMM_WORLD, &req[reqCount++]); /*epy*/
            MPI_Irecv(bepz[mybid].data(), blockSize, MPI_DOUBLE, recvFrom,
                      tag++, MPI_COMM_WORLD, &req[reqCount++]); /*epz*/
            MPI_Irecv(bew[mybid].data(), blockSize, MPI_DOUBLE, recvFrom, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*ew*/
            /*if (settings.verbose >= 2)
            {
                std::cout << "Reader rank: " << rank << " Wait for " << reqCount
                          << " requests with tag = " << tag - 8 << std::endl;
            }
            MPI_Waitall(reqCount, req, statuses);
            reqCount = 0;*/
            ++mybid;
        }

        if (settings.verbose >= 2)
        {
            std::cout << "Reader rank: " << rank << " waitall "
                      << " reqCount = " << reqCount << std::endl;
        }
        MPI_Waitall(reqCount, req, statuses);

        /* Copy 3D received blocks into 3D variable */
        mybid = 0;
        for (int b = 0; b < decomp.nblocks3D; ++b)
        {
            const auto &block = decomp.blocks3D[b];
            if (block.readerRank != rank)
            {
                continue;
            }

            /* Copy blocks into the reader's global arrays */
            Copy3D(Bx, decomp.readers[rank], bBx[mybid], block, b);
            Copy3D(By, decomp.readers[rank], bBy[mybid], block, b);
            Copy3D(Bz, decomp.readers[rank], bBz[mybid], block, b);
            Copy3D(Ex, decomp.readers[rank], bEx[mybid], block, b);
            Copy3D(Ey, decomp.readers[rank], bEy[mybid], block, b);
            Copy3D(Ez, decomp.readers[rank], bEz[mybid], block, b);
            Copy3D(jx, decomp.readers[rank], bjx[mybid], block, b);
            Copy3D(jy, decomp.readers[rank], bjy[mybid], block, b);
            Copy3D(jz, decomp.readers[rank], bjz[mybid], block, b);
            Copy3D(rho, decomp.readers[rank], brho[mybid], block, b);
            ++mybid;
        }

        /* Copy 1D received blocks into 1D variable */
        mybid = 0;
        for (int b = 0; b < decomp.nblocks1D; ++b)
        {
            const auto &block = decomp.blocks1D[b];
            if (block.readerRank != rank)
            {
                continue;
            }
            if (settings.verbose >= 2)
            {
                std::cout << "Reader rank: " << rank << " copy 1D block " << b
                          << " blockSize = " << block.count
                          << " offset = " << block.start - d.start1D
                          << std::endl;
            }
            /* Copy blocks into the reader's global arrays */
            std::copy(beid[mybid].begin(), beid[mybid].end(),
                      eid.begin() + block.start - d.start1D);
            std::copy(bemx[mybid].begin(), bemx[mybid].end(),
                      emx.begin() + block.start - d.start1D);
            std::copy(bemy[mybid].begin(), bemy[mybid].end(),
                      emy.begin() + block.start - d.start1D);
            std::copy(bemz[mybid].begin(), bemz[mybid].end(),
                      emz.begin() + block.start - d.start1D);
            std::copy(bepx[mybid].begin(), bepx[mybid].end(),
                      epx.begin() + block.start - d.start1D);
            std::copy(bepy[mybid].begin(), bepy[mybid].end(),
                      epy.begin() + block.start - d.start1D);
            std::copy(bepz[mybid].begin(), bepz[mybid].end(),
                      epz.begin() + block.start - d.start1D);
            std::copy(bew[mybid].begin(), bew[mybid].end(),
                      ew.begin() + block.start - d.start1D);
            ++mybid;
        }

        if (settings.readerDump)
        {
            /* Dump data to disk */
            dump.BeginStep();
            dump.Put(vBx, Bx.data());
            dump.Put(vBy, By.data());
            dump.Put(vBz, Bz.data());
            dump.Put(vEx, Ex.data());
            dump.Put(vEy, Ey.data());
            dump.Put(vEz, Ez.data());
            dump.Put(vjx, jx.data());
            dump.Put(vjy, jy.data());
            dump.Put(vjz, jz.data());
            dump.Put(vrho, rho.data());
            dump.Put(veid, eid.data());
            dump.Put(vemx, emx.data());
            dump.Put(vemy, emy.data());
            dump.Put(vemz, emz.data());
            dump.Put(vepx, epx.data());
            dump.Put(vepy, epy.data());
            dump.Put(vepz, epz.data());
            dump.Put(vew, ew.data());
            dump.EndStep();
        }
    }
    if (settings.readerDump)
    {
        dump.Close();
    }
}

/* Copy a block into the larger variable array
   The block is assumed to be completely inside the array */
void IO_MPI::Copy3D(std::vector<double> &var3d, const struct ReaderDecomp &vd,
                    const std::vector<double> &blockdata,
                    const struct Block3D &block, const int blockid)
{
    static int msgs = 0;
    const int msglimit = 1;
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
        for (size_t x = 0; x < end[0]; ++x)
        {
            size_t bcount = block.count[1] * block.count[2];
            size_t boffset = x * block.count[1] * block.count[2];
            size_t voffset = (start[0] + x) * vd.count3D[1] * vd.count3D[2] +
                             start[1] * vd.count3D[2];
            std::copy(blockdata.begin() + boffset,
                      blockdata.begin() + boffset + bcount - 1,
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
        for (size_t x = 0; x < end[0]; ++x)
        {
            for (size_t y = 0; y < end[1]; ++y)
            {
                size_t bcount = block.count[2];
                size_t boffset =
                    x * block.count[1] * block.count[2] + y * block.count[2];
                size_t voffset =
                    (start[0] + x) * vd.count3D[1] * vd.count3D[2] +
                    (start[1] + y) * vd.count3D[2];
                std::copy(blockdata.begin() + boffset,
                          blockdata.begin() + boffset + bcount - 1,
                          var3d.begin() + voffset);
            }
        }
    }
    ++msgs;
}
