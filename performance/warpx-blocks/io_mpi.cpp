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
    ExchangeWorldRanks(isWriter);
}

/* Global communication */
void IO_MPI::ExchangeWorldRanks(bool isWriter)
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

void IO_MPI::WriterMPI()
{
    /* pre-calculate how many blocks we have to write to manage all
       MPI requests at once */
    size_t nMyBlocks3D = 0;
    for (int b = 0; b < decomp.nblocks3D; ++b)
    {
        if (decomp.blocks3D[b].writerRank == rank)
        {
            ++nMyBlocks3D;
        }
    }
    size_t nMyBlocks1D = 0;
    for (int b = 0; b < decomp.nblocks1D; ++b)
    {
        if (decomp.blocks1D[b].writerRank == rank)
        {
            ++nMyBlocks1D;
        }
    }
    MPI_Request req[10 * nMyBlocks3D + 8 * nMyBlocks1D];
    MPI_Status statuses[10 * nMyBlocks3D + 8 * nMyBlocks1D];
    if (settings.verbose >= 2)
    {
        std::cout << "Writer rank: " << rank << " will have "
                  << 10 * nMyBlocks3D + 8 * nMyBlocks1D << " requests "
                  << std::endl;
    }

    // const WriterDecomp &d = decomp.writers[rank];

    for (int step = 0; step < settings.steps; ++step)
    {
        if (!rank)
        {
            std::cout << "Writer Step: " << step << std::endl;
        }

        int reqCount = 0;

        /* 3D variables */
#if 0
        for (int b = 0; b < decomp.nblocks3D; ++b)
        {
            const auto &block = decomp.blocks3D[b];
            if (block.writerRank != rank)
            {
                continue;
            }

            size_t blockSize = block.count[0] * block.count[1] * block.count[2];
            std::vector<double> data3d(blockSize);
            double value = rank + step / 100.0;
            for (size_t i = 0; i < blockSize; ++i)
            {
                data3d[i] = value;
            }

            int sendTo = readerWorldRanks[block.readerRank];
            int tag = b * 100;
            MPI_Request req;
            MPI_Isend(data3d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req); /*Bx*/
            MPI_Isend(data3d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req); /*By*/
            MPI_Isend(data3d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req); /*Bz*/
            MPI_Isend(data3d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req); /*Ex*/
            MPI_Isend(data3d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req); /*Ey*/
            MPI_Isend(data3d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req); /*Ez*/
            MPI_Isend(data3d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req); /*jx*/
            MPI_Isend(data3d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req); /*jy*/
            MPI_Isend(data3d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req); /*jz*/
            MPI_Isend(data3d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req); /*rho*/
        }
#endif

        /* 1D variables */
        for (int b = 0; b < decomp.nblocks1D; ++b)
        {
            const auto &block = decomp.blocks1D[b];
            if (block.writerRank != rank)
            {
                continue;
            }

            size_t blockSize = block.count;
            std::vector<double> data1d(blockSize);
            double value = 10 + rank + step / 100.0;
            for (size_t i = 0; i < blockSize; ++i)
            {
                data1d[i] = value;
            }

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
            //            MPI_Send(data1d.data(), blockSize, MPI_DOUBLE, sendTo,
            //            tag++,
            //                     MPI_COMM_WORLD); /*eid*/

            MPI_Isend(data1d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*eid*/
            MPI_Isend(data1d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*emx*/
            MPI_Isend(data1d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*emy*/
            MPI_Isend(data1d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*emz*/
            MPI_Isend(data1d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*epx*/
            MPI_Isend(data1d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*epy*/
            MPI_Isend(data1d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*epz*/
            MPI_Isend(data1d.data(), blockSize, MPI_DOUBLE, sendTo, tag++,
                      MPI_COMM_WORLD, &req[reqCount++]); /*ew*/
            if (settings.verbose >= 2)
            {
                std::cout << "Writer rank: " << rank << " Wait for " << reqCount
                          << " requests with tag = " << tag - 8
                          << " reqCount = " << reqCount << std::endl;
            }
            MPI_Waitall(reqCount, req, statuses);
            reqCount = 0;
        }

        /*if (settings.verbose >= 2)
        {
            std::cout << "Writer rank: " << rank << " waitall "
                      << " reqCount = " << reqCount << std::endl;
        }
        MPI_Waitall(reqCount, req, statuses);*/
    }
}

void IO_MPI::ReaderMPI()
{
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

    /*
     *  Pre-allocate all blocks that will come in
     */

    /* Pre-allocate individual 3D block variables received from writers */
    std::vector<double> bBx, bBy, bBz, bEx, bEy, bEz, bjx, bjy, bjz, brho;

    /* Pre-allocate individual 1D block variables received from writers */
    size_t nMyBlocks1D = 0;
    std::vector<std::vector<double>> beid, bemx, bemy, bemz, bepx, bepy, bepz,
        bew;
    for (int b = 0; b < decomp.nblocks1D; ++b)
    {
        if (decomp.blocks1D[b].readerRank == rank)
        {
            ++nMyBlocks1D;
        }
    }
    beid.resize(nMyBlocks1D);
    bemx.resize(nMyBlocks1D);
    bemy.resize(nMyBlocks1D);
    bemz.resize(nMyBlocks1D);
    bepx.resize(nMyBlocks1D);
    bepy.resize(nMyBlocks1D);
    bepz.resize(nMyBlocks1D);
    bew.resize(nMyBlocks1D);
    size_t mybid = 0;
    for (int b = 0; b < decomp.nblocks1D; ++b)
    {
        const auto &block = decomp.blocks1D[b];
        if (block.readerRank == rank)
        {
            beid[mybid].resize(block.count);
            bemx[mybid].resize(block.count);
            bemy[mybid].resize(block.count);
            bemz[mybid].resize(block.count);
            bepx[mybid].resize(block.count);
            bepy[mybid].resize(block.count);
            bepz[mybid].resize(block.count);
            bew[mybid].resize(block.count);
            ++mybid;
        }
    }

    adios2::Box<adios2::Dims> sel3D = {
        {d.start3D[0], d.start3D[1], d.start3D[2]},
        {d.count3D[0], d.count3D[1], d.count3D[2]}};

    adios2::Box<adios2::Dims> sel1D = {{d.start1D}, {d.count1D}};

    adios2::ADIOS adios(settings.adios_config, comm);
    adios2::IO io = adios.DeclareIO("WarpX");
    adios2::Engine dump;
    if (settings.readerDump)
    {
        io.SetEngine("FileStream");
        dump = io.Open("dump_mpi.bp", adios2::Mode::Write);
    }
    adios2::Dims shape1d{decomp.shape1D};
    adios2::Dims start1d{d.start1D};
    adios2::Dims count1d{d.count1D};
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

        /* 3D variables */

        /* 1D variables */
        MPI_Request req[8 * nMyBlocks1D];
        MPI_Status statuses[8 * nMyBlocks1D];
        int reqCount = 0;
        if (settings.verbose >= 2)
        {
            std::cout << "Reader rank: " << rank << " will have "
                      << 8 * nMyBlocks1D << " requests " << std::endl;
        }

        mybid = 0;
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

            // MPI_Recv(beid[mybid].data(), blockSize, MPI_DOUBLE, recvFrom,
            //          tag++, MPI_COMM_WORLD, &statuses[0]); /*eid*/
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
            if (settings.verbose >= 2)
            {
                std::cout << "Reader rank: " << rank << " Wait for " << reqCount
                          << " requests with tag = " << tag - 8 << std::endl;
            }
            MPI_Waitall(reqCount, req, statuses);
            reqCount = 0;
            ++mybid;
        }

        /*if (settings.verbose >= 2)
        {
            std::cout << "Reader rank: " << rank << " waitall "
                      << " reqCount = " << reqCount << std::endl;
        }
        MPI_Waitall(reqCount, req, statuses);*/

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

            /*dump.Put(vBx, Bx.data());
            dump.Put(vBy, By.data());
            dump.Put(vBz, Bz.data());
            dump.Put(vEx, Ex.data());
            dump.Put(vEy, Ey.data());
            dump.Put(vEz, Ez.data());
            dump.Put(vjx, jx.data());
            dump.Put(vjy, jy.data());
            dump.Put(vjz, jz.data());
            dump.Put(vrho, rho.data());*/
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
