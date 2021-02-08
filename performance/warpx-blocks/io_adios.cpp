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

void writerADIOS(const WarpxSettings &settings, const Decomp &decomp,
                 MPI_Comm comm)
{
    std::string inputFileName;
    std::string variableName;
    std::string variableType;

    int rank, nproc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    adios2::ADIOS adios(settings.adios_config, comm);
    adios2::IO io = adios.DeclareIO("WarpX");

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

    for (int step = 0; step < settings.steps; ++step)
    {
        if (!rank)
        {
            std::cout << "Writer Step: " << step << std::endl;
        }
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
            engine.Put(vBx, data3d.data(), adios2::Mode::Sync);
            engine.Put(vBy, data3d.data(), adios2::Mode::Sync);
            engine.Put(vBz, data3d.data(), adios2::Mode::Sync);
            engine.Put(vEx, data3d.data(), adios2::Mode::Sync);
            engine.Put(vEy, data3d.data(), adios2::Mode::Sync);
            engine.Put(vEz, data3d.data(), adios2::Mode::Sync);
            engine.Put(vjx, data3d.data(), adios2::Mode::Sync);
            engine.Put(vjy, data3d.data(), adios2::Mode::Sync);
            engine.Put(vjz, data3d.data(), adios2::Mode::Sync);
            engine.Put(vrho, data3d.data(), adios2::Mode::Sync);
        }

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
            engine.Put(veid, data1d.data(), adios2::Mode::Sync);
            engine.Put(vemx, data1d.data(), adios2::Mode::Sync);
            engine.Put(vemy, data1d.data(), adios2::Mode::Sync);
            engine.Put(vemz, data1d.data(), adios2::Mode::Sync);
            engine.Put(vepx, data1d.data(), adios2::Mode::Sync);
            engine.Put(vepy, data1d.data(), adios2::Mode::Sync);
            engine.Put(vepz, data1d.data(), adios2::Mode::Sync);
            engine.Put(vew, data1d.data(), adios2::Mode::Sync);
        }

        engine.EndStep();
    }
    engine.Close();
}

void readerADIOS(const WarpxSettings &settings, const Decomp &decomp,
                 MPI_Comm comm)
{
    std::string inputFileName;
    std::string variableName;
    std::string variableType;

    int rank, nproc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    const ReaderDecomp &d = decomp.readers[rank];
    std::vector<double> Bx(d.nElems3D), By(d.nElems3D), Bz(d.nElems3D),
        Ex(d.nElems3D), Ey(d.nElems3D), Ez(d.nElems3D), jx(d.nElems3D),
        jy(d.nElems3D), jz(d.nElems3D), rho(d.nElems3D);

    std::vector<double> eid(d.count1D), emx(d.count1D), emy(d.count1D),
        emz(d.count1D), epx(d.count1D), epy(d.count1D), epz(d.count1D),
        ew(d.count1D);

    adios2::Box<adios2::Dims> sel3D = {
        {d.start3D[0], d.start3D[1], d.start3D[2]},
        {d.count3D[0], d.count3D[1], d.count3D[2]}};

    adios2::Box<adios2::Dims> sel1D = {{d.start1D}, {d.count1D}};

    adios2::ADIOS adios(settings.adios_config, comm);
    adios2::IO io = adios.DeclareIO("WarpX");

    adios2::Engine engine = io.Open(settings.streamName, adios2::Mode::Read);

    adios2::Engine dump;
    if (settings.readerDump)
    {
        io.SetEngine("FileStream");
        dump = io.Open("dump_adios.bp", adios2::Mode::Write);
    }

    for (int step = 0; step < settings.steps; ++step)
    {
        if (!rank)
        {
            std::cout << "Reader Step: " << step << std::endl;
        }
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
        engine.Get(vBx, Bx.data());
        engine.Get(vBy, By.data());
        engine.Get(vBz, Bz.data());
        engine.Get(vEx, Ex.data());
        engine.Get(vEy, Ey.data());
        engine.Get(vEz, Ez.data());
        engine.Get(vjx, jx.data());
        engine.Get(vjy, jy.data());
        engine.Get(vjz, jz.data());
        engine.Get(vrho, rho.data());

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
        engine.Get(veid, eid.data());
        engine.Get(vemx, emx.data());
        engine.Get(vemy, emy.data());
        engine.Get(vemz, emz.data());
        engine.Get(vepx, epx.data());
        engine.Get(vepy, epy.data());
        engine.Get(vepz, epz.data());
        engine.Get(vew, ew.data());

        engine.EndStep();

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
    engine.Close();
    if (settings.readerDump)
    {
        dump.Close();
    }
}
