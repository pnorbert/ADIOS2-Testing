#ifndef IO_POSIX_H_
#define IO_POSIX_H_

#include <mpi.h>

#include "decomp.h"
#include "io.h"
#include "warpxsettings.h"

// align processes to this page size
constexpr size_t alignBytes = 65536;
// 2GB limit on posix IO calls
constexpr size_t DefaultMaxFileBatchSize = 2147381248;

struct AggrInfo
{
    size_t subfileIdx;    // rank's subfile
    size_t offset;        // rank's starting offset in subfile
    size_t stepSize;      // rank's size to write each step with alignment
    size_t stepTotalSize; // size of subfile per step with alignment
};

class IO_POSIX : public IO
{
public:
    IO_POSIX(const WarpxSettings &settings, const Decomp &decomp, MPI_Comm comm,
             const bool isWriter);
    ~IO_POSIX() = default;

    Timers Writer();
    Timers Reader();

private:
    /* Each process writes to one subfile, with a group of processes */
    int aggrSubfileIdx;
    MPI_Comm aggrComm; // procs writing together one file
    int aggrRank, aggrNProc;
    size_t aggrOffset;        // my starting offset in subfile
    size_t aggrStepSize;      // my size to write each step
    size_t aggrStepTotalSize; // size of subfile per step

    std::string SysErrMsg(int errnum) const;

    /*
        Writer only functions
    */
    void Write(int fd, const void *data, const size_t size);
    void SetupAggregation();
    void WriteAggregationInfo();

    /*
        Reader only functions
    */
    std::vector<AggrInfo> ReadAggregationInfo();
    std::vector<int> OpenSubfiles(const std::vector<AggrInfo> &aggrInfo);
    void CloseSubfiles(std::vector<int> &fds);
    void Read(int fd, void *data, const size_t size);
};

#endif /* IO_POSIX_H_ */
