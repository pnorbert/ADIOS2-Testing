#ifndef IO_MPI_H_
#define IO_MPI_H_

#include <mpi.h>

#include "decomp.h"
#include "io.h"
#include "warpxsettings.h"

class IO_MPI : public IO
{
public:
    IO_MPI(const WarpxSettings &settings, const Decomp &decomp, MPI_Comm comm,
           const bool isWriter);
    ~IO_MPI() = default;

    Timers Writer();
    Timers Reader();

private:
    int worldRank, worldNProc;
    std::vector<int> readerWorldRanks;
    std::vector<int> writerWorldRanks;
    void ExchangeWorldRanks();
};

#endif /* IO_MPI_H_ */
