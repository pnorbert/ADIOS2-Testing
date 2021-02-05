#ifndef IO_ADIOS_H_
#define IO_ADIOS_H_

#include <mpi.h>

#include "warpxsettings.h"
#include "decomp.h"

void writerADIOS(const WarpxSettings &settings, const Decomp &decomp,
                 MPI_Comm comm);
void readerADIOS(const WarpxSettings &settings, const Decomp &decomp,
                 MPI_Comm comm);

#endif /* IO_ADIOS_H_ */
