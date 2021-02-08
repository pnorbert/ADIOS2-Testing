#ifndef DECOMP_H_
#define DECOMP_H_

#include <cstddef>
#include <mpi.h>

#include "warpxsettings.h"

struct Block1D
{
    size_t producerID;
    size_t start;
    size_t count;
    int writerRank;
    int readerRank;
};

struct Block3D
{
    size_t producerID;
    size_t start[3];
    size_t count[3];
    int writerRank;
    int readerRank;
};

struct ReaderDecomp
{
    /* Reader decomposition calculated from shape1D and shape3D */
    size_t start1D, count1D;
    size_t start3D[3], count3D[3];
    /* reader rank position in each dimension */
    size_t pos3D[3]; /* rp[i]=0..settings.readerDecomp3D[i]-1 */
    size_t nElems3D; /* Prod(count3D) */
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
    int nProducers; /* original number of writers */

    struct Block1D *blocks1D;
    struct Block3D *blocks3D;

    std::vector<ReaderDecomp> readers;

private:
    const WarpxSettings &s;
    const MPI_Comm comm;
    int rank, nproc;
    std::string BroadcastFile(const std::string &fileName, MPI_Comm comm) const;
    void ProcessDecomp1D(const std::string &fileContent);
    void ProcessDecomp3D(const std::string &fileContent);
    void DecomposeWriters();
    void DecomposeReaders1D();
    void DecomposeReaders3D();
    void CalculateReceivers3D();
};

#endif /* DECOMP_H_ */
