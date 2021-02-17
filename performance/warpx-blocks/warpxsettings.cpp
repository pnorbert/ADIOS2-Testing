#include "warpxsettings.h"

#include <fstream>
#include <iostream>

#include "json.hpp"

void to_json(nlohmann::json &j, const WarpxSettings &s)
{
    std::string mode;
    if (s.ioMode == IOMode::MPI)
    {
        mode = "MPI";
    }
    else
    {
        mode = "ADIOS";
    }

    double cpt = s.computeTime.count();
    j = nlohmann::json{{"ioMode", mode},
                       {"steps", s.steps},
                       {"computeTime", cpt},
                       {"input1D", s.inputfile1D},
                       {"input3D", s.inputfile3D},
                       {"streamName", s.streamName},
                       {"adios_config", s.adios_config},
                       {"nWriters", s.nWriters},
                       {"readDecomp3D", s.readDecomp3D},
                       {"posixAggregatorRatio", s.posixAggregatorRatio},
                       {"readerDump", s.readerDump},
                       {"adiosLockSelections", s.adiosLockSelections},
                       {"verbose", s.verbose}};
}

void from_json(const nlohmann::json &j, WarpxSettings &s)
{
    std::string mode;
    double cpt;
    j.at("ioMode").get_to(mode);
    j.at("steps").get_to(s.steps);
    j.at("computeTime").get_to(cpt);
    j.at("input1D").get_to(s.inputfile1D);
    j.at("input3D").get_to(s.inputfile3D);
    j.at("streamName").get_to(s.streamName);
    j.at("adios_config").get_to(s.adios_config);
    j.at("nWriters").get_to(s.nWriters);
    j.at("readDecomp3D").get_to(s.readDecomp3D);
    j.at("readerDump").get_to(s.readerDump);
    j.at("adiosLockSelections").get_to(s.adiosLockSelections);
    j.at("posixAggregatorRatio").get_to(s.posixAggregatorRatio);
    j.at("verbose").get_to(s.verbose);

    s.computeTime = Seconds(cpt);

    std::string modestr = mode;
    std::transform(modestr.begin(), modestr.end(), modestr.begin(), ::tolower);
    if (modestr == "mpi")
    {
        s.ioMode = IOMode::MPI;
    }
    else if (modestr == "adios")
    {
        s.ioMode = IOMode::ADIOS;
    }
    else if (modestr == "posix")
    {
        s.ioMode = IOMode::POSIX;
    }
    else
    {
        std::cout << "Invalid ioMode argument:" << mode
                  << ". Reverting to POSIX mode..." << std::endl;
        s.ioMode = IOMode::POSIX;
    }

    if (s.posixAggregatorRatio == 0)
    {
        std::cout << "Invalid 'posixAggregatorRatio' argument:"
                  << s.posixAggregatorRatio << ". Reverting to value 1 ..."
                  << std::endl;
        s.posixAggregatorRatio = 1;
    }

    s.nReaders = s.readDecomp3D[0] * s.readDecomp3D[1] * s.readDecomp3D[2];
}

WarpxSettings::WarpxSettings() {}

WarpxSettings WarpxSettings::from_json(const std::string &fname)
{
    std::ifstream ifs(fname);
    nlohmann::json j;

    ifs >> j;

    return j.get<WarpxSettings>();
}
