#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <string>
#include <vector>

enum class CouplingMode { MPI, ADIOS };

struct Settings {
    CouplingMode cplMode;
    int steps;
    std::string inputfile1D;
    std::string inputfile3D;
    std::string streamName;
    std::string adios_config;
    std::vector<size_t> readDecomp3D;
    

    Settings();
    static Settings from_json(const std::string &fname);
};

#endif
