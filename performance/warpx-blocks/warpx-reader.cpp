#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <algorithm>
#include <vector>
#include <list>
#include <sstream>
#include <random>
#include <unordered_map>

#include <mpi.h>
#include "adios2.h"

#include "decomp.h"
#include "settings.h"

enum class InputMode { MPI, ADIOS };

int rank, nproc;
MPI_Comm app_comm;
InputMode mode = InputMode::ADIOS;
int nsteps = 0;

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " settings.json"  << std::endl;
}


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int wrank, wnproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wnproc);
    const unsigned int color = 2;
    MPI_Comm_split(MPI_COMM_WORLD, color, wrank, &app_comm);
    MPI_Comm_rank(app_comm, &rank);
    MPI_Comm_size(app_comm, &nproc);

    /* Process arguments */
    if (argc < 2)
    {
        std::cout << "# of arguments provided: " << argc << ", not enough arguments" << std::endl;
        show_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    const std::string settingsFileName = argv[1];

    Settings settings = Settings::from_json(argv[1]);
    
    /* Process input specs decomp_3D.in and decomp_1D.in */
    Decomp decomp(settings.inputfile1D, settings.inputfile3D, app_comm);

    if (!rank) {
        std::cout << "reader decomposition 3D: {" << settings.readDecomp3D[0] << ", " 
                  << settings.readDecomp3D[1]  << ", " << settings.readDecomp3D[2] << "}"
                  << std::endl;
        std::cout << "input 1D: " << settings.inputfile1D
                  << "\ninput 3D: " << settings.inputfile3D
                  << "\nstream name: " << settings.streamName
                  << "\nsteps: " << settings.steps
                  << "\nadios config: " << settings.adios_config
                  << std::endl;
    }

    size_t n = settings.readDecomp3D[0] * settings.readDecomp3D[1] * settings.readDecomp3D[2];
    if (n != static_cast<size_t>(nproc))
    {
        std::cout << "Reader decomposition is invalid for " << nproc << " processes." 
                  << " The readDecomp3D should divide up the number of readers."
                  << std::endl;
        MPI_Finalize();
        return 1;
    }

    
    MPI_Finalize();

    return 0;
}
