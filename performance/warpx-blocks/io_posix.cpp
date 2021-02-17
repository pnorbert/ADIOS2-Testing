#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <fcntl.h> // open
#include <fstream>
#include <iostream>
#include <stddef.h> // write output
#include <string>
#include <sys/stat.h>  // open, fstat
#include <sys/types.h> // open
#include <unistd.h>    // write, close
#include <vector>

#include "io_posix.h"

#include "adios2.h"

IO_POSIX::IO_POSIX(const WarpxSettings &settings, const Decomp &decomp,
                   MPI_Comm comm, const bool isWriter)
: IO(settings, decomp, comm, isWriter)
{
    if (isWriter)
    {
        SetupAggregation();
        WriteAggregationInfo();
    }
}

void IO_POSIX::SetupAggregation()
{
    aggrSubfileIdx = rank / settings.posixAggregatorRatio;
    MPI_Comm_split(comm, aggrSubfileIdx, rank, &aggrComm);
    MPI_Comm_rank(aggrComm, &aggrRank);
    MPI_Comm_size(aggrComm, &aggrNProc);

    std::vector<size_t> sizes(aggrNProc);
    size_t pad = (alignBytes - nTotalAllocatedSize % alignBytes) % alignBytes;
    aggrStepSize = nTotalAllocatedSize + pad;

    MPI_Allgather(&aggrStepSize, sizeof(size_t), MPI_BYTE, sizes.data(),
                  sizeof(size_t), MPI_BYTE, aggrComm);

    aggrOffset = 0;
    aggrStepTotalSize = 0;
    for (int i = 0; i < aggrRank; ++i)
    {
        aggrOffset += sizes[i];
        aggrStepTotalSize += sizes[i];
    }
    for (int i = aggrRank; i < aggrNProc; ++i)
    {
        aggrStepTotalSize += sizes[i];
    }
}

std::string IO_POSIX::SysErrMsg(int errnum) const
{
    return std::string(": errno = " + std::to_string(errnum) + ": " +
                       strerror(errnum));
}

void IO_POSIX::Write(int fd, const void *data, const size_t size)
{
    const char *buffer = static_cast<const char *>(data);
    auto lf_Write = [&](int fd, const char *buffer, size_t size) {
        while (size > 0)
        {
            errno = 0;
            const auto writtenSize = write(fd, buffer, size);
            if (writtenSize == -1)
            {
                if (errno == EINTR)
                {
                    continue;
                }

                throw std::ios_base::failure("ERROR: couldn't write to file: " +
                                             SysErrMsg(errno));
            }

            buffer += writtenSize;
            size -= writtenSize;
        }
    };

    if (size > DefaultMaxFileBatchSize)
    {
        const size_t batches = size / DefaultMaxFileBatchSize;
        const size_t remainder = size % DefaultMaxFileBatchSize;

        size_t position = 0;
        for (size_t b = 0; b < batches; ++b)
        {
            lf_Write(fd, &buffer[position], DefaultMaxFileBatchSize);
            position += DefaultMaxFileBatchSize;
        }
        lf_Write(fd, &buffer[position], remainder);
    }
    else
    {
        lf_Write(fd, buffer, size);
    }
}

void IO_POSIX::Read(int fd, void *data, const size_t size)
{
    char *buffer = static_cast<char *>(data);
    auto lf_Read = [&](int fd, char *buffer, size_t size) {
        while (size > 0)
        {
            errno = 0;
            const auto readSize = read(fd, buffer, size);
            if (readSize == -1)
            {
                if (errno == EINTR)
                {
                    continue;
                }

                throw std::ios_base::failure(
                    "ERROR: couldn't read from file: " + SysErrMsg(errno));
            }

            buffer += readSize;
            size -= readSize;
        }
    };

    if (size > DefaultMaxFileBatchSize)
    {
        const size_t batches = size / DefaultMaxFileBatchSize;
        const size_t remainder = size % DefaultMaxFileBatchSize;

        size_t position = 0;
        for (size_t b = 0; b < batches; ++b)
        {
            lf_Read(fd, &buffer[position], DefaultMaxFileBatchSize);
            position += DefaultMaxFileBatchSize;
        }
        lf_Read(fd, &buffer[position], remainder);
    }
    else
    {
        lf_Read(fd, buffer, size);
    }
}

void IO_POSIX::WriteAggregationInfo()
{
    AggrInfo info;
    info.subfileIdx = aggrSubfileIdx;
    info.offset = aggrOffset;
    info.stepSize = aggrStepSize;
    info.stepTotalSize = aggrStepTotalSize;

    std::vector<AggrInfo> aggrInfo;
    if (!rank)
    {
        aggrInfo.resize(nproc);
    }

    MPI_Gather(&info, sizeof(AggrInfo), MPI_BYTE, aggrInfo.data(),
               sizeof(AggrInfo), MPI_BYTE, 0, comm);

    if (!rank)
    {
        std::string fname = "warpx.idx";
        errno = 0;
        int fd = open(fname.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
        if (fd == -1)
        {
            std::cout << "ERROR: Writer rank: " << rank
                      << " could not create file " << fname << " error code "
                      << errno << " : " << strerror(errno) << std::endl;
            MPI_Abort(comm, errno);
        }

        write(fd, aggrInfo.data(), nproc * sizeof(AggrInfo));
        close(fd);
    }
}

std::vector<AggrInfo> IO_POSIX::ReadAggregationInfo()
{
    std::vector<AggrInfo> aggrInfo(settings.nWriters);
    const size_t nBytes =
        static_cast<ssize_t>(settings.nWriters * sizeof(AggrInfo));
    if (!rank)
    {
        std::string fname = "warpx.idx";
        errno = 0;
        int fd = open(fname.c_str(), O_RDONLY);
        if (fd == -1)
        {
            std::cout << "ERROR: Reader rank: " << rank
                      << " could not open file " << fname << " error code "
                      << errno << " : " << strerror(errno) << std::endl;
            MPI_Abort(comm, errno);
        }

        const auto n = read(fd, aggrInfo.data(), nBytes);
        if (n != static_cast<ssize_t>(nBytes))
        {
            std::cout << "ERROR: Reader rank: " << rank << " could not read "
                      << nBytes << " bytes from file " << fname << std::endl;
            MPI_Abort(comm, errno);
        }
        close(fd);
    }

    MPI_Bcast(aggrInfo.data(), nBytes, MPI_BYTE, 0, comm);
    return aggrInfo;
}

Timers IO_POSIX::Writer()
{
    Timers t;
    TimePoint ts, te;
    TimePoint totalstart = std::chrono::steady_clock::now();

    if (settings.verbose >= 1)
    {
        std::cout << "Writer rank: " << rank << " will have "
                  << 10 * nMyBlocks3D + 8 * nMyBlocks1D
                  << " blocks of total size " << nTotalAllocatedSize
                  << " bytes padded to " << aggrStepSize << " bytes to subfile "
                  << aggrSubfileIdx << " and subrank " << aggrRank << " of "
                  << aggrNProc << " processes writing " << aggrStepTotalSize
                  << " bytes per step" << std::endl;
    }

    std::string fname = "warpx." + std::to_string(aggrSubfileIdx);
    errno = 0;
    int fd = open(fname.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (fd == -1)
    {
        std::cout << "ERROR: Writer rank: " << rank << " could not create file "
                  << fname << " error code " << errno << " : "
                  << strerror(errno) << std::endl;
        MPI_Abort(comm, errno);
    }

    for (int step = 1; step <= settings.steps; ++step)
    {
        if (!rank)
        {
            std::cout << "Writer Step: " << step << std::endl;
        }

        ts = std::chrono::steady_clock::now();
        Compute(step);
        te = std::chrono::steady_clock::now();
        t.compute += te - ts;
        ts = te;

        const size_t myOffset = aggrOffset + (step - 1) * aggrStepTotalSize;
        lseek64(fd, myOffset, SEEK_SET);

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
            if (settings.verbose >= 2)
            {
                std::cout << "Writer rank: " << rank << " write 3D blocks " << b
                          << " blockSize = " << blockSize << std::endl;
            }

            Write(fd, bBx[mybid].data(), blockSize * sizeof(double));
            Write(fd, bBy[mybid].data(), blockSize * sizeof(double));
            Write(fd, bBz[mybid].data(), blockSize * sizeof(double));
            Write(fd, bEx[mybid].data(), blockSize * sizeof(double));
            Write(fd, bEy[mybid].data(), blockSize * sizeof(double));
            Write(fd, bEz[mybid].data(), blockSize * sizeof(double));
            Write(fd, bjx[mybid].data(), blockSize * sizeof(double));
            Write(fd, bjy[mybid].data(), blockSize * sizeof(double));
            Write(fd, bjz[mybid].data(), blockSize * sizeof(double));
            Write(fd, brho[mybid].data(), blockSize * sizeof(double));
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
            if (settings.verbose >= 2)
            {
                std::cout << "Writer rank: " << rank << " isend 1D block " << b
                          << " blockSize = " << blockSize << std::endl;
            }

            Write(fd, beid[mybid].data(), blockSize * sizeof(double));
            Write(fd, bemx[mybid].data(), blockSize * sizeof(double));
            Write(fd, bemy[mybid].data(), blockSize * sizeof(double));
            Write(fd, bemz[mybid].data(), blockSize * sizeof(double));
            Write(fd, bepx[mybid].data(), blockSize * sizeof(double));
            Write(fd, bepy[mybid].data(), blockSize * sizeof(double));
            Write(fd, bepz[mybid].data(), blockSize * sizeof(double));
            Write(fd, bew[mybid].data(), blockSize * sizeof(double));
            ++mybid;
        }
        te = std::chrono::steady_clock::now();
        t.output += te - ts;
    }

    close(fd);
    t.total = std::chrono::steady_clock::now() - totalstart;
    return t;
}

std::vector<int> IO_POSIX::OpenSubfiles(const std::vector<AggrInfo> &aggrInfo)
{
    std::vector<int> fds;
    size_t nfiles = settings.nWriters / settings.posixAggregatorRatio;
    if (settings.nWriters % settings.posixAggregatorRatio)
    {
        ++nfiles;
    }
    fds.resize(nfiles, 0);
    for (int b = 0; b < decomp.nblocks3D; ++b)
    {
        const auto &block = decomp.blocks3D[b];
        if (block.readerRank != rank)
        {
            continue;
        }
        const int subfile = aggrInfo[block.writerRank].subfileIdx;
        if (fds[subfile] == 0)
        {
            errno = 0;
            std::string fname = "warpx." + std::to_string(subfile);
            fds[subfile] = open(fname.c_str(), O_RDONLY);
            if (fds[subfile] == -1)
            {
                std::cout << "ERROR: Reader rank: " << rank
                          << " could not open file " << fname << " error code "
                          << errno << " : " << strerror(errno) << std::endl;
                MPI_Abort(comm, errno);
            }
        }
    }
    for (int b = 0; b < decomp.nblocks1D; ++b)
    {
        const auto &block = decomp.blocks1D[b];
        if (block.readerRank != rank)
        {
            continue;
        }
        const int subfile = aggrInfo[block.writerRank].subfileIdx;
        if (fds[subfile] == 0)
        {
            errno = 0;
            std::string fname = "warpx." + std::to_string(subfile);
            fds[subfile] = open(fname.c_str(), O_RDONLY);
            if (fds[subfile] == -1)
            {
                std::cout << "ERROR: Reader rank: " << rank
                          << " could not open file " << fname << " error code "
                          << errno << " : " << strerror(errno) << std::endl;
                MPI_Abort(comm, errno);
            }
        }
    }
    return fds;
}

void IO_POSIX::CloseSubfiles(std::vector<int> &fds)
{
    for (const auto fd : fds)
    {
        close(fd);
    }
}

Timers IO_POSIX::Reader()
{
    Timers t;
    TimePoint ts, te;
    TimePoint totalstart = std::chrono::steady_clock::now();

    if (settings.verbose >= 2)
    {
        std::cout << "Reader rank: " << rank << " will have "
                  << 10 * nMyBlocks3D + 8 * nMyBlocks1D << " requests "
                  << std::endl;
    }

    const ReaderDecomp &d = decomp.readers[rank];
    /* Reader's variables in memory */
    std::vector<double> Bx(d.nElems3D), By(d.nElems3D), Bz(d.nElems3D),
        Ex(d.nElems3D), Ey(d.nElems3D), Ez(d.nElems3D), jx(d.nElems3D),
        jy(d.nElems3D), jz(d.nElems3D), rho(d.nElems3D);

    std::vector<double> eid(d.count1D), emx(d.count1D), emy(d.count1D),
        emz(d.count1D), epx(d.count1D), epy(d.count1D), epz(d.count1D),
        ew(d.count1D);

    ts = std::chrono::steady_clock::now();
    std::vector<AggrInfo> aggrInfo = ReadAggregationInfo();
    std::vector<int> fds = OpenSubfiles(aggrInfo);
    te = std::chrono::steady_clock::now();
    t.input += te - ts;
    ts = te;

    adios2::ADIOS adios(settings.adios_config, comm);
    adios2::IO io = adios.DeclareIO("WarpX");
    adios2::Engine dump;
    if (settings.readerDump)
    {
        io.SetEngine("FileStream");
        dump = io.Open("dump_posix.bp", adios2::Mode::Write);
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
    te = std::chrono::steady_clock::now();
    t.output += te - ts;

    std::vector<size_t> offsets(fds.size());

    for (int step = 1; step <= settings.steps; ++step)
    {
        if (!rank)
        {
            std::cout << "Reader Step: " << step << std::endl;
        }

        ts = std::chrono::steady_clock::now();
        size_t mybid = 0;
        std::fill(offsets.begin(), offsets.end(), 0);

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

            const auto &a = aggrInfo[block.writerRank];
            const int subfd = fds[a.subfileIdx];

            size_t offset = (step - 1) * a.stepTotalSize + a.offset;
            if (offsets[a.subfileIdx] == 0)
            {
                lseek64(subfd, offset, SEEK_SET);
                offsets[a.subfileIdx] = offset;
            }
            if (settings.verbose >= 2)
            {
                std::cout << "Reader rank: " << rank << " read ten 3D blocks "
                          << b << " blockSize = " << blockSize
                          << " from writer " << block.writerRank
                          << " from subfile " << a.subfileIdx
                          << " offset = " << offset << std::endl;
            }

            Read(subfd, bBx[mybid].data(), blockSize);  /*Bx*/
            Read(subfd, bBy[mybid].data(), blockSize);  /*By*/
            Read(subfd, bBz[mybid].data(), blockSize);  /*Bz*/
            Read(subfd, bEx[mybid].data(), blockSize);  /*Ex*/
            Read(subfd, bEy[mybid].data(), blockSize);  /*Ey*/
            Read(subfd, bEz[mybid].data(), blockSize);  /*Ez*/
            Read(subfd, bjx[mybid].data(), blockSize);  /*jx*/
            Read(subfd, bjy[mybid].data(), blockSize);  /*jy*/
            Read(subfd, bjz[mybid].data(), blockSize);  /*jz*/
            Read(subfd, brho[mybid].data(), blockSize); /*rho*/

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

            const auto &a = aggrInfo[block.writerRank];
            const int subfd = fds[a.subfileIdx];

            size_t offset = (step - 1) * a.stepTotalSize + a.offset;
            lseek64(subfd, offset, SEEK_SET);

            if (settings.verbose >= 2)
            {
                std::cout << "Reader rank: " << rank << " read eight 1D blocks "
                          << b << " blockSize = " << blockSize
                          << " from writer " << block.writerRank
                          << " from subfile "
                          << aggrInfo[block.writerRank].subfileIdx
                          << " offset = " << offset << std::endl;
            }

            Read(subfd, beid[mybid].data(), blockSize); /*eid*/
            Read(subfd, bemx[mybid].data(), blockSize); /*emx*/
            Read(subfd, bemy[mybid].data(), blockSize); /*emy*/
            Read(subfd, bemz[mybid].data(), blockSize); /*emz*/
            Read(subfd, bepx[mybid].data(), blockSize); /*epx*/
            Read(subfd, bepy[mybid].data(), blockSize); /*epy*/
            Read(subfd, bepz[mybid].data(), blockSize); /*epz*/
            Read(subfd, bew[mybid].data(), blockSize);  /*ew*/

            ++mybid;
        }

        te = std::chrono::steady_clock::now();
        t.input += te - ts;
        ts = te;

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

        te = std::chrono::steady_clock::now();
        t.compute += te - ts;
        ts = te;

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
            te = std::chrono::steady_clock::now();
            t.output += te - ts;
        }
    }
    if (settings.readerDump)
    {
        ts = std::chrono::steady_clock::now();
        dump.Close();
        te = std::chrono::steady_clock::now();
        t.output += te - ts;
    }

    ts = std::chrono::steady_clock::now();
    CloseSubfiles(fds);
    te = std::chrono::steady_clock::now();
    t.input += te - ts;
    ts = te;

    t.total = std::chrono::steady_clock::now() - totalstart;
    return t;
}
