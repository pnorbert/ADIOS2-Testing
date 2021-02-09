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
    void ExchangeWorldRanks();

    size_t nMyBlocks1D = 0;
    size_t nMyBlocks3D = 0;
    void CalculateMyBlocks();

    /* Individual 3D block variables sent by / received from writers */
    std::vector<std::vector<double>> bBx, bBy, bBz, bEx, bEy, bEz, bjx, bjy,
        bjz, brho;

    /* Individual 1D block variables sent by / received from writers */
    std::vector<std::vector<double>> beid, bemx, bemy, bemz, bepx, bepy, bepz,
        bew;
    void AllocateBlocks();

    void Copy3D(std::vector<double> &, const struct ReaderDecomp &,
                const std::vector<double> &, const struct Block3D &,
                const int blockid);
};

#endif /* IO_MPI_H_ */
