#ifndef DECOMP_H_
#define DECOMP_H_

#include <cstddef>
#include <mpi.h>

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

    Decomp(const std::string &inputfile1D, const std::string &inputfile3D, MPI_Comm comm);
    ~Decomp();
    
    size_t shape1D;
    size_t shape3D[3];

    int nblocks1D;
    int nblocks3D;
    int nWriters; /* original number of writers */

    struct Block1D * blocks1D;
    struct Block3D * blocks3D;

    /* each process takes care of a range of writerIDs */
    size_t minWriterID;
    size_t maxWriterID;

private:

    const MPI_Comm comm;
    int rank, nproc;
    std::string BroadcastFile(const std::string &fileName, MPI_Comm comm) const;
    void ProcessDecomp1D(const std::string &fileContent);
    void ProcessDecomp3D(const std::string &fileContent);
    void DecomposeWriters();
};

#endif /* DECOMP_H_ */

