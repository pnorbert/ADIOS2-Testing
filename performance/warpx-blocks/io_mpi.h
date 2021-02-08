#ifndef IO_MPI_H_
#define IO_MPI_H_

#include <mpi.h>

#include "decomp.h"
#include "warpxsettings.h"

class IO_MPI
{
public:
    IO_MPI(const WarpxSettings &settings, const Decomp &decomp, MPI_Comm comm,
           const bool isWriter);
    ~IO_MPI() = default;

    void WriterMPI();
    void ReaderMPI();

private:
    const WarpxSettings &settings;
    const Decomp &decomp;
    const MPI_Comm comm;
    const bool isWriter;
    int rank, nproc;
    int worldRank, worldNProc;
    std::vector<int> readerWorldRanks;
    std::vector<int> writerWorldRanks;
    void ExchangeWorldRanks(bool isWriter);
};

#endif /* IO_MPI_H_ */
