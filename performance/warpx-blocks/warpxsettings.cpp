#include "warpxsettings.h"

#include <fstream>
#include <iostream>

#include "json.hpp"

void to_json(nlohmann::json &j, const WarpxSettings &s)
{
    std::string cpl;
    if (s.cplMode == CouplingMode::MPI)
    {
        cpl = "MPI";
    }
    else
    {
        cpl = "ADIOS";
    }

    double cpt = s.computeTime.count();
    j = nlohmann::json{{"couplingMode", cpl},
                       {"steps", s.steps},
                       {"computeTime", cpt},
                       {"input1D", s.inputfile1D},
                       {"input3D", s.inputfile3D},
                       {"streamName", s.streamName},
                       {"adios_config", s.adios_config},
                       {"nWriters", s.nWriters},
                       {"readDecomp3D", s.readDecomp3D},
                       {"readerDump", s.readerDump},
                       {"verbose", s.verbose}};
}

void from_json(const nlohmann::json &j, WarpxSettings &s)
{
    std::string cpl;
    double cpt;
    j.at("couplingMode").get_to(cpl);
    j.at("steps").get_to(s.steps);
    j.at("computeTime").get_to(cpt);
    j.at("input1D").get_to(s.inputfile1D);
    j.at("input3D").get_to(s.inputfile3D);
    j.at("streamName").get_to(s.streamName);
    j.at("adios_config").get_to(s.adios_config);
    j.at("nWriters").get_to(s.nWriters);
    j.at("readDecomp3D").get_to(s.readDecomp3D);
    j.at("readerDump").get_to(s.readerDump);
    j.at("verbose").get_to(s.verbose);

    s.computeTime = Seconds(cpt);

    std::string modestr = cpl;
    std::transform(modestr.begin(), modestr.end(), modestr.begin(), ::tolower);
    if (modestr == "mpi")
    {
        s.cplMode = CouplingMode::MPI;
    }
    else if (modestr == "adios")
    {
        s.cplMode = CouplingMode::ADIOS;
    }
    else
    {
        std::cout << "Invalid couplingMode argument:" << cpl
                  << ". Reverting to MPI mode..." << std::endl;
        s.cplMode = CouplingMode::MPI;
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
