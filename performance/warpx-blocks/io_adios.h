#ifndef IO_ADIOS_H_
#define IO_ADIOS_H_

#include <mpi.h>

#include "decomp.h"
#include "io.h"
#include "warpxsettings.h"

class IO_ADIOS : public IO
{
public:
    IO_ADIOS(const WarpxSettings &settings, const Decomp &decomp, MPI_Comm comm,
             const bool isWriter);
    ~IO_ADIOS() = default;

    Timers Writer();
    Timers Reader();
};

#endif /* IO_ADIOS_H_ */
