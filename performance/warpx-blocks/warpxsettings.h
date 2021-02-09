#ifndef __SETTINGS_H__
#define __SETTINGS_H__

#include <chrono>
#include <string>
#include <vector>

enum class CouplingMode
{
    MPI,
    ADIOS
};

typedef std::chrono::duration<double> Seconds;
typedef std::chrono::time_point<
    std::chrono::steady_clock,
    std::chrono::duration<double, std::chrono::steady_clock::period>>
    TimePoint;

struct WarpxSettings
{
    CouplingMode cplMode;
    int steps;
    Seconds computeTime;
    std::string inputfile1D;
    std::string inputfile3D;
    std::string streamName;
    std::string adios_config;
    size_t nWriters;
    std::vector<size_t> readDecomp3D;
    bool readerDump;
    size_t verbose; /* 0-2 */

    WarpxSettings();
    static WarpxSettings from_json(const std::string &fname);

    size_t nReaders; /* calculated from readDecomp3D */
};

#endif
