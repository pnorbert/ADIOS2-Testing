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

enum class OutputMode { MPI, ADIOS };

int rank, nproc;
MPI_Comm app_comm;
OutputMode mode = OutputMode::ADIOS;
int nsteps = 0;

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " output_file  mode  steps\n"
              << " mode:  MPI | ADIOS"
              << " steps: 0 or more"
              << std::endl;
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
    if (argc < 4)
    {
        std::cout << "# of arguments provided: " << argc << ", not enough arguments" << std::endl;
        show_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }
    const std::string outputFileName = argv[1];
    std::string modestr = argv[2];
    std::transform(modestr.begin(), modestr.end(), modestr.begin(), ::tolower);
    if (modestr == "mpi")
    {
        mode = OutputMode::MPI;
    }
    else if (modestr == "adios")
    {
        mode = OutputMode::ADIOS;
    }
    else
    {
        std::cout << "Invalid mode argument:" << std::endl;
        show_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    try
    {
        const int out = std::stoi(argv[3]);
        nsteps = out;
    }
    catch (...)
    {
        std::cout << "Invalid steps argument:" << std::endl;
        show_usage(argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* Process input specs decomp_3D.in and decomp_1D.in */
    Decomp decomp(app_comm);


/*
    std::string inputFileName;
    std::string variableName;
    std::string variableType;

    {
        adios2::ADIOS adios("adios2.xml", MPI_COMM_WORLD);

        adios2::IO reader_io = adios.DeclareIO("ReaderIO");
        adios2::Engine reader_engine =
            reader_io.Open(inputFileName, adios2::Mode::Read);

        adios2::IO writer_io = adios.DeclareIO("WriterIO");
        writer_io.SetEngine("BP4");
        adios2::Engine writer_engine =
            writer_io.Open(outputFileName, adios2::Mode::Write);
        const std::map<std::string, adios2::Params> allVariables =
            reader_io.AvailableVariables();

        for (const auto variablePair : allVariables)
        {
            std::string variableName;
            std::string variableType;
            variableName = variablePair.first;
            variableType = variablePair.second.at("Type");
            if (!variableName.empty())
            {
                if (variableType == "double")
                {
                    auto variable = reader_io.InquireVariable<double>(variableName);
                    //auto blocksInfo = reader.BlocksInfo(variable, targetStep);
                    auto blocksInfo = reader_engine.AllStepsBlocksInfo(variable).at(targetStep);
                    size_t spaceDimensions = variable.Shape().size();
                    std::cout << spaceDimensions << "D variable " << variableName << " has " << blocksInfo.size()
                              << " blocks in step " << targetStep << std::endl;

                    auto newVariable =
                        writer_io.DefineVariable<double>(variableName, variable.Shape());

                    for (const auto &info : blocksInfo)
                    {
                        // std::cout << "    block " << info.BlockID
                        //           << " offset = " << DimsToString(info.Start)
                        //           << " size = " << DimsToString(info.Count)
                        //           << " writer ID = " << info.WriterID
                        //           << std::endl;
                        if (rank != info.WriterID)
                        {
                            continue;
                        }
                        size_t obSize = 1;
                        for (size_t i = 0; i < info.Count.size(); i++)
                        {
                            obSize *= info.Count[i];
                        }
                        std::vector<double> obData(obSize);
                        variable.SetSelection({info.Start, info.Count});
                        variable.SetStepSelection({targetStep, 1});
                        reader_engine.Get(variable, obData.data(), adios2::Mode::Sync);
                        newVariable.SetSelection({info.Start, info.Count});
                        writer_engine.Put(newVariable, obData.data(), adios2::Mode::Sync);
                    }
                }
                else if (variableType == "uint64_t")
                {
                    auto variable = reader_io.InquireVariable<uint64_t>(variableName);
                    auto blocksInfo = reader_engine.AllStepsBlocksInfo(variable).at(targetStep);
                    size_t spaceDimensions = variable.Shape().size();
                    std::cout << spaceDimensions << "D variable " << variableName << " has " << blocksInfo.size()
                              << " blocks in step " << targetStep << std::endl;

                    auto newVariable =
                        writer_io.DefineVariable<uint64_t>(variableName, variable.Shape());

                    for (const auto &info : blocksInfo)
                    {
                        // std::cout << "    block " << info.BlockID
                        //           << " offset = " << DimsToString(info.Start)
                        //           << " size = " << DimsToString(info.Count)
                        //           << " writer ID = " << info.WriterID
                        //           << std::endl;
                        if (rank != info.WriterID)
                        {
                            continue;
                        }
                        size_t obSize = 1;
                        for (size_t i = 0; i < info.Count.size(); i++)
                        {
                            obSize *= info.Count[i];
                        }
                        std::vector<uint64_t> obData(obSize);
                        variable.SetSelection({info.Start, info.Count});
                        variable.SetStepSelection({targetStep, 1});
                        reader_engine.Get(variable, obData.data(), adios2::Mode::Sync);
                        newVariable.SetSelection({info.Start, info.Count});
                        writer_engine.Put(newVariable, obData.data(), adios2::Mode::Sync);
                    }
                }
            }
        }

        reader_engine.Close();
        writer_engine.Close();
    }
*/
    MPI_Finalize();

    return 0;
}
