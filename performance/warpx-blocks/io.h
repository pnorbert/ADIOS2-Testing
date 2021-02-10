#ifndef IO_H_
#define IO_H_

#include <mpi.h>

#include "decomp.h"
#include "timers.h"
#include "warpxsettings.h"

class IO
{
public:
    IO(const WarpxSettings &settings, const Decomp &decomp, MPI_Comm comm,
       const bool isWriter);
    ~IO() = default;

    virtual Timers Writer() = 0;
    virtual Timers Reader() = 0;

protected:
    const WarpxSettings &settings;
    const Decomp &decomp;
    const MPI_Comm comm;
    const bool isWriter;
    int rank, nproc;

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

    /*
        Writer only functions
    */
    void Compute(const int step);
};

#endif /* IO_MPI_H_ */
