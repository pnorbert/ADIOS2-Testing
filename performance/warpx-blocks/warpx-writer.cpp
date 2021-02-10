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

#include <mpi.h>

#include "decomp.h"
#include "io_adios.h"
#include "io_mpi.h"
#include "timers.h"
#include "warpxsettings.h"

int rank, nproc;
MPI_Comm app_comm;

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " settings.json" << std::endl;
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

    WarpxSettings settings = WarpxSettings::from_json(argv[1]);

    if (settings.nWriters != static_cast<size_t>(nproc))
    {
        std::cout << "Writer size is invalid. Number of processes " << nproc
                  << " must equal to \"nWriter\": " << settings.nWriters
                  << " in the settings file." << std::endl;
        MPI_Finalize();
        return 1;
    }

    /* Process input specs decomp_3D.in and decomp_1D.in */
    Decomp decomp(settings, app_comm);

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

    Timers t;
    if (settings.cplMode == CouplingMode::ADIOS)
    {
        IO_ADIOS io(settings, decomp, app_comm, true);
        t = io.Writer();
    }
    else // (settings.cplMode == CouplingMode::MPI)
    {
        IO_MPI io(settings, decomp, app_comm, true);
        t = io.Writer();
    }

    std::vector<Timers> tv = GatherTimers(t, app_comm, rank, nproc);
    struct Timers ta = AvgTimes(tv);
    if (!rank)
    {
        std::cout << "Writer timing: Average compute = " << ta.compute.count()
                  << " input = " << ta.input.count()
                  << " output = " << ta.output.count() << std::endl;
    }

    MPI_Finalize();
    return 0;
}
