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

#include "adios2.h"
#include <mpi.h>

#include "io_adios.h"

IO_ADIOS::IO_ADIOS(const WarpxSettings &settings, const Decomp &decomp,
                   MPI_Comm comm, const bool isWriter)
: IO(settings, decomp, comm, isWriter, isWriter)
{
}

Timers IO_ADIOS::Writer()
{
    Timers t;
    TimePoint ts, te;
    TimePoint totalstart = std::chrono::steady_clock::now();

    adios2::ADIOS adios(settings.adios_config, comm);
    adios2::IO io = adios.DeclareIO("WarpX");
    io.SetParameter("InitialBufferSize",
                    std::to_string(1.1 * nTotalAllocatedSize));

    adios2::Engine engine = io.Open(settings.streamName, adios2::Mode::Write);

    adios2::Dims start3d{0, 0, 0};
    adios2::Dims count3d{1, 1, 1};
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
    adios2::Dims start1d{0};
    adios2::Dims count1d{1};
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

        int mybid = 0;

        engine.BeginStep();

        /* 3D variables */
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

            adios2::Dims count3d(block.count, block.count + 3);
            adios2::Dims start3d(block.start, block.start + 3);
            vBx.SetSelection({start3d, count3d});
            vBy.SetSelection({start3d, count3d});
            vBz.SetSelection({start3d, count3d});
            vEx.SetSelection({start3d, count3d});
            vEy.SetSelection({start3d, count3d});
            vEz.SetSelection({start3d, count3d});
            vjx.SetSelection({start3d, count3d});
            vjy.SetSelection({start3d, count3d});
            vjz.SetSelection({start3d, count3d});
            vrho.SetSelection({start3d, count3d});
            engine.Put(vBx, bBx[mybid].data());
            engine.Put(vBy, bBy[mybid].data());
            engine.Put(vBz, bBz[mybid].data());
            engine.Put(vEx, bEx[mybid].data());
            engine.Put(vEy, bEy[mybid].data());
            engine.Put(vEz, bEz[mybid].data());
            engine.Put(vjx, bjx[mybid].data());
            engine.Put(vjy, bjy[mybid].data());
            engine.Put(vjz, bjz[mybid].data());
            engine.Put(vrho, brho[mybid].data());

            ++mybid;
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
            std::vector<double> data1d(blockSize);
            double value = rank + step / 100.0;
            for (size_t i = 0; i < blockSize; ++i)
            {
                data1d[i] = value;
            }

            veid.SetSelection({{block.start}, {block.count}});
            vemx.SetSelection({{block.start}, {block.count}});
            vemy.SetSelection({{block.start}, {block.count}});
            vemz.SetSelection({{block.start}, {block.count}});
            vepx.SetSelection({{block.start}, {block.count}});
            vepy.SetSelection({{block.start}, {block.count}});
            vepz.SetSelection({{block.start}, {block.count}});
            vew.SetSelection({{block.start}, {block.count}});
            engine.Put(veid, beid[mybid].data());
            engine.Put(vemx, bemx[mybid].data());
            engine.Put(vemy, bemy[mybid].data());
            engine.Put(vemz, bemz[mybid].data());
            engine.Put(vepx, bepx[mybid].data());
            engine.Put(vepy, bepy[mybid].data());
            engine.Put(vepz, bepz[mybid].data());
            engine.Put(vew, bew[mybid].data());

            ++mybid;
        }

        if (step == 1 && settings.adiosLockSelections)
        {
            engine.LockWriterDefinitions();
        }

        engine.EndStep();
        te = std::chrono::steady_clock::now();
        t.output += te - ts;
    }

    ts = std::chrono::steady_clock::now();
    engine.Close();
    te = std::chrono::steady_clock::now();
    t.output += te - ts;
    t.total = std::chrono::steady_clock::now() - totalstart;
    return t;
}

Timers IO_ADIOS::Reader()
{
    if (settings.readerDump)
    {
        return ReaderWithSpan();
    }
    else
    {
        return ReaderNoDump();
    }
}

Timers IO_ADIOS::ReaderWithSpan()
{
    Timers t;
    TimePoint ts, te;
    TimePoint totalstart = std::chrono::steady_clock::now();

    const ReaderDecomp &d = decomp.readers[rank];

    size_t outputSize =
        10 * d.nElems3D * sizeof(double) + 8 * d.count1D * sizeof(double);

    adios2::Box<adios2::Dims> sel3D = {
        {d.start3D[0], d.start3D[1], d.start3D[2]},
        {d.count3D[0], d.count3D[1], d.count3D[2]}};

    adios2::Box<adios2::Dims> sel1D = {{d.start1D}, {d.count1D}};

    adios2::ADIOS adios(settings.adios_config, comm);
    adios2::IO io = adios.DeclareIO("WarpX");
    io.SetParameter("InitialBufferSize", std::to_string(outputSize + 4194304));
    // if (!rank)
    {
        std::cout << "Reader rank " << rank
                  << " allocated small blocks size = " << nTotalAllocatedSize
                  << " output size = " << outputSize << std::endl;
    }

    adios2::Engine engine = io.Open(settings.streamName, adios2::Mode::Read);

    adios2::Engine dump;
    if (settings.readerDump)
    {
        io.SetEngine("FileStream");
    }
    else
    {
        io.SetEngine("NullCore");
    }
    dump = io.Open("dump_adios.bp", adios2::Mode::Write);

    for (int step = 1; step <= settings.steps; ++step)
    {
        if (!rank)
        {
            std::cout << "Reader Step: " << step << std::endl;
        }

        ts = std::chrono::steady_clock::now();
        engine.BeginStep();
        dump.BeginStep();

        /* 3D variables */

        adios2::Variable<double> vBx =
            io.InquireVariable<double>("/data/fields/B/x");
        adios2::Variable<double> vBy =
            io.InquireVariable<double>("/data/fields/B/y");
        adios2::Variable<double> vBz =
            io.InquireVariable<double>("/data/fields/B/z");
        adios2::Variable<double> vEx =
            io.InquireVariable<double>("/data/fields/E/x");
        adios2::Variable<double> vEy =
            io.InquireVariable<double>("/data/fields/E/y");
        adios2::Variable<double> vEz =
            io.InquireVariable<double>("/data/fields/E/z");
        adios2::Variable<double> vjx =
            io.InquireVariable<double>("/data/fields/j/x");
        adios2::Variable<double> vjy =
            io.InquireVariable<double>("/data/fields/j/y");
        adios2::Variable<double> vjz =
            io.InquireVariable<double>("/data/fields/j/z");
        adios2::Variable<double> vrho =
            io.InquireVariable<double>("/data/fields/rho");

        vBx.SetSelection(sel3D);
        vBy.SetSelection(sel3D);
        vBz.SetSelection(sel3D);
        vEx.SetSelection(sel3D);
        vEy.SetSelection(sel3D);
        vEz.SetSelection(sel3D);
        vjx.SetSelection(sel3D);
        vjy.SetSelection(sel3D);
        vjz.SetSelection(sel3D);
        vrho.SetSelection(sel3D);

        /* Use adios2::Span to get memory from output buffer */
        adios2::Variable<double>::Span sBx = dump.Put(vBx);
        adios2::Variable<double>::Span sBy = dump.Put(vBy);
        adios2::Variable<double>::Span sBz = dump.Put(vBz);
        adios2::Variable<double>::Span sEx = dump.Put(vEx);
        adios2::Variable<double>::Span sEy = dump.Put(vEy);
        adios2::Variable<double>::Span sEz = dump.Put(vEz);
        adios2::Variable<double>::Span sjx = dump.Put(vjx);
        adios2::Variable<double>::Span sjy = dump.Put(vjy);
        adios2::Variable<double>::Span sjz = dump.Put(vjz);
        adios2::Variable<double>::Span srho = dump.Put(vrho);

        /* Populate spans by reading in data */
        engine.Get(vBx, sBx.data(), adios2::Mode::Sync);
        engine.Get(vBy, sBy.data(), adios2::Mode::Sync);
        engine.Get(vBz, sBz.data(), adios2::Mode::Sync);
        engine.Get(vEx, sEx.data(), adios2::Mode::Sync);
        engine.Get(vEy, sEy.data(), adios2::Mode::Sync);
        engine.Get(vEz, sEz.data(), adios2::Mode::Sync);
        engine.Get(vjx, sjx.data(), adios2::Mode::Sync);
        engine.Get(vjy, sjy.data(), adios2::Mode::Sync);
        engine.Get(vjz, sjz.data(), adios2::Mode::Sync);
        engine.Get(vrho, srho.data(), adios2::Mode::Sync);

        /* 1D variables */

        adios2::Variable<double> veid =
            io.InquireVariable<double>("/data/800/particles/electrons/id");
        adios2::Variable<double> vemx = io.InquireVariable<double>(
            "/data/800/particles/electrons/momentum/x");
        adios2::Variable<double> vemy = io.InquireVariable<double>(
            "/data/800/particles/electrons/momentum/y");
        adios2::Variable<double> vemz = io.InquireVariable<double>(
            "/data/800/particles/electrons/momentum/z");
        adios2::Variable<double> vepx = io.InquireVariable<double>(
            "/data/800/particles/electrons/position/x");
        adios2::Variable<double> vepy = io.InquireVariable<double>(
            "/data/800/particles/electrons/position/y");
        adios2::Variable<double> vepz = io.InquireVariable<double>(
            "/data/800/particles/electrons/position/z");
        adios2::Variable<double> vew = io.InquireVariable<double>(
            "/data/800/particles/electrons/weighting");

        veid.SetSelection(sel1D);
        vemx.SetSelection(sel1D);
        vemy.SetSelection(sel1D);
        vemz.SetSelection(sel1D);
        vepx.SetSelection(sel1D);
        vepy.SetSelection(sel1D);
        vepz.SetSelection(sel1D);
        vew.SetSelection(sel1D);

        /* Use adios2::Span to get memory from output buffer */
        adios2::Variable<double>::Span seid = dump.Put(veid);
        adios2::Variable<double>::Span semx = dump.Put(vemx);
        adios2::Variable<double>::Span semy = dump.Put(vemy);
        adios2::Variable<double>::Span semz = dump.Put(vemz);
        adios2::Variable<double>::Span sepx = dump.Put(vepx);
        adios2::Variable<double>::Span sepy = dump.Put(vepy);
        adios2::Variable<double>::Span sepz = dump.Put(vepz);
        adios2::Variable<double>::Span sew = dump.Put(vew);

        /* Populate spans by reading in data */
        engine.Get(veid, seid.data(), adios2::Mode::Sync);
        engine.Get(vemx, semx.data(), adios2::Mode::Sync);
        engine.Get(vemy, semy.data(), adios2::Mode::Sync);
        engine.Get(vemz, semz.data(), adios2::Mode::Sync);
        engine.Get(vepx, sepx.data(), adios2::Mode::Sync);
        engine.Get(vepy, sepy.data(), adios2::Mode::Sync);
        engine.Get(vepz, sepz.data(), adios2::Mode::Sync);
        engine.Get(vew, sew.data(), adios2::Mode::Sync);

        if (step == 1 && settings.adiosLockSelections)
        {
            engine.LockReaderSelections();
            dump.LockWriterDefinitions();
        }

        engine.EndStep();
        te = std::chrono::steady_clock::now();
        t.input += te - ts;
        ts = te;

        /* Dump data to disk */
        dump.EndStep();
        te = std::chrono::steady_clock::now();
        t.output += te - ts;
    }
    ts = std::chrono::steady_clock::now();
    engine.Close();
    te = std::chrono::steady_clock::now();
    t.input += te - ts;
    te = ts;

    dump.Close();
    te = std::chrono::steady_clock::now();
    t.output += te - ts;

    t.total = std::chrono::steady_clock::now() - totalstart;
    return t;
}

Timers IO_ADIOS::ReaderNoDump()
{
    Timers t;
    TimePoint ts, te;
    TimePoint totalstart = std::chrono::steady_clock::now();

    const ReaderDecomp &d = decomp.readers[rank];
    std::vector<double> Bx(d.nElems3D), By(d.nElems3D), Bz(d.nElems3D),
        Ex(d.nElems3D), Ey(d.nElems3D), Ez(d.nElems3D), jx(d.nElems3D),
        jy(d.nElems3D), jz(d.nElems3D), rho(d.nElems3D);

    std::vector<double> eid(d.count1D), emx(d.count1D), emy(d.count1D),
        emz(d.count1D), epx(d.count1D), epy(d.count1D), epz(d.count1D),
        ew(d.count1D);

    size_t outputSize =
        10 * d.nElems3D * sizeof(double) + 8 * d.count1D * sizeof(double);

    adios2::Box<adios2::Dims> sel3D = {
        {d.start3D[0], d.start3D[1], d.start3D[2]},
        {d.count3D[0], d.count3D[1], d.count3D[2]}};

    adios2::Box<adios2::Dims> sel1D = {{d.start1D}, {d.count1D}};

    adios2::ADIOS adios(settings.adios_config, comm);
    adios2::IO io = adios.DeclareIO("WarpX");
    io.SetParameter("InitialBufferSize", std::to_string(outputSize + 4194304));
    // if (!rank)
    {
        std::cout << "Reader rank " << rank
                  << " allocated small blocks size = " << nTotalAllocatedSize
                  << " output size = " << outputSize << std::endl;
    }

    adios2::Engine engine = io.Open(settings.streamName, adios2::Mode::Read);

    adios2::Engine dump;
    io.SetEngine("FileStream");

    for (int step = 1; step <= settings.steps; ++step)
    {
        if (!rank)
        {
            std::cout << "Reader Step: " << step << std::endl;
        }

        ts = std::chrono::steady_clock::now();
        engine.BeginStep();

        /* 3D variables */

        adios2::Variable<double> vBx =
            io.InquireVariable<double>("/data/fields/B/x");
        adios2::Variable<double> vBy =
            io.InquireVariable<double>("/data/fields/B/y");
        adios2::Variable<double> vBz =
            io.InquireVariable<double>("/data/fields/B/z");
        adios2::Variable<double> vEx =
            io.InquireVariable<double>("/data/fields/E/x");
        adios2::Variable<double> vEy =
            io.InquireVariable<double>("/data/fields/E/y");
        adios2::Variable<double> vEz =
            io.InquireVariable<double>("/data/fields/E/z");
        adios2::Variable<double> vjx =
            io.InquireVariable<double>("/data/fields/j/x");
        adios2::Variable<double> vjy =
            io.InquireVariable<double>("/data/fields/j/y");
        adios2::Variable<double> vjz =
            io.InquireVariable<double>("/data/fields/j/z");
        adios2::Variable<double> vrho =
            io.InquireVariable<double>("/data/fields/rho");

        vBx.SetSelection(sel3D);
        vBy.SetSelection(sel3D);
        vBz.SetSelection(sel3D);
        vEx.SetSelection(sel3D);
        vEy.SetSelection(sel3D);
        vEz.SetSelection(sel3D);
        vjx.SetSelection(sel3D);
        vjy.SetSelection(sel3D);
        vjz.SetSelection(sel3D);
        vrho.SetSelection(sel3D);

        engine.Get(vBx, Bx.data(), adios2::Mode::Sync);
        engine.Get(vBy, By.data(), adios2::Mode::Sync);
        engine.Get(vBz, Bz.data(), adios2::Mode::Sync);
        engine.Get(vEx, Ex.data(), adios2::Mode::Sync);
        engine.Get(vEy, Ey.data(), adios2::Mode::Sync);
        engine.Get(vEz, Ez.data(), adios2::Mode::Sync);
        engine.Get(vjx, jx.data(), adios2::Mode::Sync);
        engine.Get(vjy, jy.data(), adios2::Mode::Sync);
        engine.Get(vjz, jz.data(), adios2::Mode::Sync);
        engine.Get(vrho, rho.data(), adios2::Mode::Sync);

        /* 1D variables */

        adios2::Variable<double> veid =
            io.InquireVariable<double>("/data/800/particles/electrons/id");
        adios2::Variable<double> vemx = io.InquireVariable<double>(
            "/data/800/particles/electrons/momentum/x");
        adios2::Variable<double> vemy = io.InquireVariable<double>(
            "/data/800/particles/electrons/momentum/y");
        adios2::Variable<double> vemz = io.InquireVariable<double>(
            "/data/800/particles/electrons/momentum/z");
        adios2::Variable<double> vepx = io.InquireVariable<double>(
            "/data/800/particles/electrons/position/x");
        adios2::Variable<double> vepy = io.InquireVariable<double>(
            "/data/800/particles/electrons/position/y");
        adios2::Variable<double> vepz = io.InquireVariable<double>(
            "/data/800/particles/electrons/position/z");
        adios2::Variable<double> vew = io.InquireVariable<double>(
            "/data/800/particles/electrons/weighting");

        veid.SetSelection(sel1D);
        vemx.SetSelection(sel1D);
        vemy.SetSelection(sel1D);
        vemz.SetSelection(sel1D);
        vepx.SetSelection(sel1D);
        vepy.SetSelection(sel1D);
        vepz.SetSelection(sel1D);
        vew.SetSelection(sel1D);

        engine.Get(veid, eid.data(), adios2::Mode::Sync);
        engine.Get(vemx, emx.data(), adios2::Mode::Sync);
        engine.Get(vemy, emy.data(), adios2::Mode::Sync);
        engine.Get(vemz, emz.data(), adios2::Mode::Sync);
        engine.Get(vepx, epx.data(), adios2::Mode::Sync);
        engine.Get(vepy, epy.data(), adios2::Mode::Sync);
        engine.Get(vepz, epz.data(), adios2::Mode::Sync);
        engine.Get(vew, ew.data(), adios2::Mode::Sync);

        if (step == 1 && settings.adiosLockSelections)
        {
            engine.LockReaderSelections();
        }

        engine.EndStep();
        te = std::chrono::steady_clock::now();
        t.input += te - ts;
        ts = te;
    }
    ts = std::chrono::steady_clock::now();
    engine.Close();
    te = std::chrono::steady_clock::now();
    t.input += te - ts;
    te = ts;

    t.total = std::chrono::steady_clock::now() - totalstart;
    return t;
}
