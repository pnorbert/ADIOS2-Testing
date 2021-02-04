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

#include "decomp.h"
#include "settings.h"

enum class OutputMode
{
    MPI,
    ADIOS
};

int rank, nproc;
MPI_Comm app_comm;
OutputMode mode = OutputMode::ADIOS;
int nsteps = 0;

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " settings.json" << std::endl;
}

void writerADIOS(const Settings &settings, const Decomp &decomp)
{
    std::string inputFileName;
    std::string variableName;
    std::string variableType;

    adios2::ADIOS adios(settings.adios_config, app_comm);
    adios2::IO io = adios.DeclareIO("WarpX");

    adios2::Engine engine = io.Open(settings.streamName, adios2::Mode::Write);

    adios2::Dims shape3d(decomp.shape3D, decomp.shape3D + 3);
    adios2::Dims start3d{0, 0, 0};
    adios2::Dims count3d{1, 1, 1};
    adios2::Variable<double> vBx = io.DefineVariable<double>(
        "/data/fields/B/x", shape3d, start3d, count3d, false);
    adios2::Variable<double> vBy = io.DefineVariable<double>(
        "/data/fields/B/y", shape3d, start3d, count3d, false);
    adios2::Variable<double> vBz = io.DefineVariable<double>(
        "/data/fields/B/z", shape3d, start3d, count3d, false);
    adios2::Variable<double> vEx = io.DefineVariable<double>(
        "/data/fields/E/x", shape3d, start3d, count3d, false);
    adios2::Variable<double> vEy = io.DefineVariable<double>(
        "/data/fields/E/y", shape3d, start3d, count3d, false);
    adios2::Variable<double> vEz = io.DefineVariable<double>(
        "/data/fields/E/z", shape3d, start3d, count3d, false);
    adios2::Variable<double> vjx = io.DefineVariable<double>(
        "/data/fields/j/x", shape3d, start3d, count3d, false);
    adios2::Variable<double> vjy = io.DefineVariable<double>(
        "/data/fields/j/y", shape3d, start3d, count3d, false);
    adios2::Variable<double> vjz = io.DefineVariable<double>(
        "/data/fields/j/z", shape3d, start3d, count3d, false);
    adios2::Variable<double> vrho = io.DefineVariable<double>(
        "/data/fields/rho", shape3d, start3d, count3d, false);

    adios2::Dims shape1d{decomp.shape1D};
    adios2::Dims start1d{0};
    adios2::Dims count1d{1};
    adios2::Variable<double> veid = io.DefineVariable<double>(
        "data/800/particles/electrons/id", shape1d, start1d, count1d, false);
    adios2::Variable<double> vemx =
        io.DefineVariable<double>("data/800/particles/electrons/momentum/x",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vemy =
        io.DefineVariable<double>("data/800/particles/electrons/momentum/y",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vemz =
        io.DefineVariable<double>("data/800/particles/electrons/momentum/z",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vepx =
        io.DefineVariable<double>("data/800/particles/electrons/position/x",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vepy =
        io.DefineVariable<double>("data/800/particles/electrons/position/y",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vepz =
        io.DefineVariable<double>("data/800/particles/electrons/position/z",
                                  shape1d, start1d, count1d, false);
    adios2::Variable<double> vew =
        io.DefineVariable<double>("data/800/particles/electrons/weighting",
                                  shape1d, start1d, count1d, false);

    for (int step = 0; step < settings.steps; ++step)
    {
        if (!rank)
        {
            std::cout << "Step: " << step << std::endl;
        }
        engine.BeginStep();
        for (int b = 0; b < decomp.nblocks3D; ++b)
        {
            const auto &block = decomp.blocks3D[b];
            if (block.writerID < decomp.minWriterID ||
                block.writerID > decomp.maxWriterID)
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
        engine.EndStep();
    }
    engine.Close();
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int wrank, wnproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wnproc);
    const unsigned int color = 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &app_comm);
    MPI_Comm_rank(app_comm, &rank);
    MPI_Comm_size(app_comm, &nproc);

    /* Process arguments */
    if (argc < 2)
    {
        std::cout << "# of arguments provided: " << argc
                  << ", not enough arguments" << std::endl;
        show_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    const std::string settingsFileName = argv[1];

    Settings settings = Settings::from_json(argv[1]);

    /* Process input specs decomp_3D.in and decomp_1D.in */
    Decomp decomp(settings.inputfile1D, settings.inputfile3D, app_comm);

    if (!rank)
    {
        std::cout << "reader decomposition 3D: {" << settings.readDecomp3D[0]
                  << ", " << settings.readDecomp3D[1] << ", "
                  << settings.readDecomp3D[2] << "}" << std::endl;
        std::cout << "input 1D: " << settings.inputfile1D
                  << "\ninput 3D: " << settings.inputfile3D
                  << "\nstream name: " << settings.streamName
                  << "\nsteps: " << settings.steps
                  << "\nadios config: " << settings.adios_config << std::endl;
    }

    if (settings.cplMode == CouplingMode::ADIOS)
    {
        writerADIOS(settings, decomp);
    }
    else // (settings.cplMode == CouplingMode::ADIOS)
    {
        /* code */
    }

    MPI_Finalize();
    return 0;
}