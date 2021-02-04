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

#include "adios2.h"

/*

   List the blocks of a variable from a BP file (
   Must be double or uint64_t variable for now)

   $ ./decomp /gpfs/alpine/world-shared/csc303/junmin/Jan2021/r2/N4/out-bp/bp-diags-k4/N4/openpmd_000800.bp  /data/800/particles/electrons/momentum/x > decomp_N4_128_1D.in
   $ ./decomp /gpfs/alpine/world-shared/csc303/junmin/Jan2021/r2/N4/out-bp/bp-diags-k4/N4/openpmd_000800.bp  /data/800/fields/B/x > decomp_N4_128_3D.in

*/

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> file var"
              << std::endl;
}


std::string DimsToString(const adios2::Dims &dimensions)
{
    std::string dimensionsString("");

    for (const auto dimension : dimensions)
    {
        dimensionsString += std::to_string(dimension) + " ";
    }
    dimensionsString.pop_back();
    return dimensionsString;
}

int main(int argc, char *argv[])
{
    int rank, size;
    rank = 0;
    size = 1;
    if (argc < 2)
    {
        std::cout << "# of arguments provided: " << argc << ", not enough arguments" << std::endl;
        show_usage(argv[0]);
        return 1;
    }
    std::string inputFileName = argv[1];
    std::string variableName = argv[2];
    std::string variableType;
    size_t targetStep = 0;


    adios2::ADIOS adios;
    adios2::IO reader_io = adios.DeclareIO("ReaderIO");
    adios2::Engine reader_engine =
           reader_io.Open(inputFileName, adios2::Mode::Read);

    const std::map<std::string, adios2::Params> allVariables =
           reader_io.AvailableVariables();
    
    for (const auto variablePair : allVariables)
    {
        std::string vName;
        std::string vType;
        vName = variablePair.first;
        vType = variablePair.second.at("Type");
        if (!vName.empty() && variableName == vName)
        {
            variableType = vType;
            break;
        }
    }

    if (!variableType.empty())
    {
        if (variableType == "double")
        {
            auto variable = reader_io.InquireVariable<double>(variableName);
            //auto blocksInfo = reader.BlocksInfo(variable, targetStep);
            auto blocksInfo = reader_engine.AllStepsBlocksInfo(variable).at(targetStep);
            size_t spaceDimensions = variable.Shape().size();
            //std::cout << "# " << spaceDimensions << "D variable " << variableName 
            //          << " has double type and has " << blocksInfo.size()
            //          << " blocks in step " << targetStep << std::endl;
            std::cout << "Shape " << DimsToString(variable.Shape()) << std::endl;
            std::cout << "Blocks " << blocksInfo.size() << std::endl;
            std::cout << "# Block  Writer  Starts   Counts" << std::endl;

            for (const auto &info : blocksInfo)
            {
                std::cout << info.BlockID
                          << " " << info.WriterID
                          << " " << DimsToString(info.Start)
                          << " " << DimsToString(info.Count) 
                          << std::endl;
            }
        }
        else if (variableType == "uint64_t")
        {
            auto variable = reader_io.InquireVariable<uint64_t>(variableName);
            auto blocksInfo = reader_engine.AllStepsBlocksInfo(variable).at(targetStep);
            size_t spaceDimensions = variable.Shape().size();
            //std::cout << "# " << spaceDimensions << "D variable " << variableName 
            //         << " has double type and has " << blocksInfo.size()
            //          << " blocks in step " << targetStep << std::endl;
            std::cout << "Shape " << DimsToString(variable.Shape()) << std::endl;
            std::cout << "Blocks " << blocksInfo.size() << std::endl;
            std::cout << "# Block  Writer  Starts   Counts" << std::endl;

            for (const auto &info : blocksInfo)
            {
                std::cout << info.BlockID
                          << " " << info.WriterID
                          << " " << DimsToString(info.Start)
                          << " " << DimsToString(info.Count) 
                          << std::endl;
            }
        }
    }

    reader_engine.Close();

    return 0;
}
