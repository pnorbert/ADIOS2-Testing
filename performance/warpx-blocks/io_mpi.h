#ifndef IO_MPI_H_
#define IO_MPI_H_

#include <mpi.h>

#include "warpxsettings.h"
#include "decomp.h"

void writerMPI(const WarpxSettings &settings, const Decomp &decomp,
               MPI_Comm comm);
void readerMPI(const WarpxSettings &settings, const Decomp &decomp,
               MPI_Comm comm);

#endif /* IO_MPI_H_ */
