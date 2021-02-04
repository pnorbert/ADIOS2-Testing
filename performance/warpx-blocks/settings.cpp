#include "settings.h"

#include <fstream>
#include <iostream>

#include "json.hpp"

void to_json(nlohmann::json &j, const Settings &s)
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
    
    j = nlohmann::json{{"couplingMode", cpl},
                       {"steps", s.steps},
                       {"input1D", s.inputfile1D},
                       {"input3D", s.inputfile3D},
                       {"streamName", s.streamName},
                       {"adios_config", s.adios_config},
                       {"readDecomp3D", s.readDecomp3D}
                      };
}

void from_json(const nlohmann::json &j, Settings &s)
{
    std::string cpl;
    j.at("couplingMode").get_to(cpl);
    j.at("steps").get_to(s.steps);
    j.at("input1D").get_to(s.inputfile1D);
    j.at("input3D").get_to(s.inputfile3D);
    j.at("streamName").get_to(s.streamName);
    j.at("adios_config").get_to(s.adios_config);
    j.at("readDecomp3D").get_to(s.readDecomp3D);

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
                  << ". Reverting to MPI mode..."
                  << std::endl;
        s.cplMode = CouplingMode::MPI;
    }
}

Settings::Settings(){}


Settings Settings::from_json(const std::string &fname)
{
    std::ifstream ifs(fname);
    nlohmann::json j;

    ifs >> j;

    return j.get<Settings>();
}
