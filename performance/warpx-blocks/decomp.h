#ifndef DECOMP_H_
#define DECOMP_H_

#include <cstddef>
#include <mpi.h>

#include "warpxsettings.h"

struct Block1D
{
    size_t writerID;
    size_t start;
    size_t count;
};

struct Block3D
{
    size_t writerID;
    size_t start[3];
    size_t count[3];
};

class Decomp
{
public:
    Decomp(const WarpxSettings &settings, MPI_Comm comm);
    ~Decomp();

    size_t shape1D;
    std::vector<size_t> shape3D;

    int nblocks1D;
    int nblocks3D;
    int nWriters; /* original number of writers */

    struct Block1D *blocks1D;
    struct Block3D *blocks3D;

    /* each process takes care of a range of writerIDs */
    size_t minWriterID;
    size_t maxWriterID;

    /* Reader decomposition calculated from shape1D and shape3D */
    size_t readerStart1D, readerCount1D;
    std::vector<size_t> readerStart3D, readerCount3D;
    /* reader rank position in each dimension */
    size_t readerPos3D[3]; /* rp[i]=0..settings.readerDecomp3D[i]-1 */
    size_t nElems3D;       /* Prod(readerCount3D) */

private:
    const MPI_Comm comm;
    int rank, nproc;
    std::string BroadcastFile(const std::string &fileName, MPI_Comm comm) const;
    void ProcessDecomp1D(const std::string &fileContent);
    void ProcessDecomp3D(const std::string &fileContent);
    void DecomposeWriters();
    void DecomposeReaders(std::vector<size_t> readDecomp3D);
};

#endif /* DECOMP_H_ */
