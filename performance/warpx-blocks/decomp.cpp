#include "decomp.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::string> split(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        if (token != "")
        {
            tokens.push_back(token);
        }
    }
    return tokens;
}

void split(const std::string &s, char delimiter, std::vector<std::string> &v)
{
    int maxn = v.size();
    int n = 0;
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        if (token != "")
        {
            v[n] = token;
            ++n;
        }
        if (n >= maxn)
        {
            break;
        }
    }
}

std::string Decomp::BroadcastFile(const std::string &fileName,
                                  MPI_Comm comm) const
{
    std::string fileContent;
    int length;
    // Read the file on rank 0 and broadcast it to everybody else
    if (!rank)
    {
        std::ifstream fileStream(fileName);
        if (!fileStream)
        {
            throw std::ios_base::failure("ERROR: file " + fileName +
                                         " not found\n");
        }

        std::ostringstream fileSS;
        fileSS << fileStream.rdbuf();
        fileStream.close();
        fileContent = fileSS.str();
        length = fileContent.size();
        // std::cout << "Length of " << fileName << " is " << length  <<
        // std::endl; std::cout << "Start: [" << fileContent.substr(0,20) << "]"
        // << std::endl;
    }

    if (nproc > 1)
    {
        MPI_Bcast(&length, 1, MPI_INT, 0, comm);
        if (rank > 0)
        {
            fileContent.resize(length);
        }
        MPI_Bcast(const_cast<char *>(fileContent.data()), length, MPI_CHAR, 0,
                  comm);
    }
    return fileContent;
}

/* Process decomp_1D.in */
void Decomp::ProcessDecomp1D(const std::string &fileContent)
{
    std::string line;
    std::istringstream fin(fileContent);

    // Shape
    getline(fin, line);
    std::vector<std::string> v(4);
    split(line, ' ', v);

    if (v[0] != "Shape")
    {
        throw std::runtime_error(
            "First non-comment line of 1D decomp file must be Shape ");
    }
    shape1D = std::stol(v[1]);

    // Blocks
    getline(fin, line);
    split(line, ' ', v);
    if (v[0] != "Blocks")
    {
        throw std::runtime_error(
            "Second non-comment line of 1D decomp file must be Blocks ");
    }
    nblocks1D = std::stol(v[1]);

    blocks1D = static_cast<Block1D *>(malloc(nblocks1D * sizeof(Block1D)));
    // Read blocks
    int blockID = 0;
    while (getline(fin, line))
    {
        if (!line.size() || line[0] == '#')
        {
            continue;
        }
        split(line, ' ', v);
        blocks1D[blockID].writerID = std::stol(v[1]);
        blocks1D[blockID].start = std::stol(v[2]);
        blocks1D[blockID].count = std::stol(v[3]);
        ++blockID;
    }

    if (!rank)
    {
        std::cout << "decomp 1D: Shape = " << shape1D
                  << "  nBlocks = " << nblocks1D << "  found = " << blockID
                  << std::endl;
    }
}

/* Process decomp_3D.in */
void Decomp::ProcessDecomp3D(const std::string &fileContent)
{
    std::string line;
    size_t maxWriterID = 0;
    std::istringstream fin(fileContent);

    // Shape
    getline(fin, line);
    std::vector<std::string> v(9);
    split(line, ' ', v);
    if (v[0] != "Shape")
    {
        throw std::runtime_error(
            "First non-comment line of 3D decomp file must be Shape ");
    }
    shape3D.push_back(std::stol(v[1]));
    shape3D.push_back(std::stol(v[2]));
    shape3D.push_back(std::stol(v[3]));

    // Blocks
    getline(fin, line);
    split(line, ' ', v);
    if (v[0] != "Blocks")
    {
        throw std::runtime_error(
            "Second non-comment line of 3D decomp file must be Blocks ");
    }
    nblocks3D = std::stol(v[1]);

    blocks3D = static_cast<Block3D *>(malloc(nblocks3D * sizeof(Block3D)));
    // Read blocks
    int blockID = 0;
    while (getline(fin, line))
    {
        if (!line.size() || line[0] == '#')
        {
            continue;
        }
        split(line, ' ', v);
        blocks3D[blockID].writerID = std::stol(v[1]);
        if (blocks3D[blockID].writerID > maxWriterID)
        {
            maxWriterID = blocks3D[blockID].writerID;
        }
        blocks3D[blockID].start[0] = std::stol(v[2]);
        blocks3D[blockID].start[1] = std::stol(v[3]);
        blocks3D[blockID].start[2] = std::stol(v[4]);
        blocks3D[blockID].count[0] = std::stol(v[5]);
        blocks3D[blockID].count[1] = std::stol(v[6]);
        blocks3D[blockID].count[2] = std::stol(v[7]);
        ++blockID;
    }

    nWriters = maxWriterID + 1;
    if (!rank)
    {
        std::cout << "decomp 3D: Shape = {" << shape3D[0] << " x " << shape3D[1]
                  << " x " << shape3D[2] << "}"
                  << "  nBlocks = " << nblocks3D << "  found = " << blockID
                  << std::endl;
        std::cout << "nWriters = " << nWriters << std::endl;
    }
}

void Decomp::DecomposeWriters()
{
    int ne = nWriters / nproc;
    minWriterID = rank * ne;
    int rem = nWriters % nproc;
    if (rank < rem)
    {
        ++ne;
        minWriterID += rank;
    }
    else
    {
        minWriterID += rem;
    }
    maxWriterID = minWriterID + ne - 1;
    std::cout << "rank " << rank << " writerIDs " << minWriterID << " - "
              << maxWriterID << std::endl;
}

void Decomp::DecomposeReaders(std::vector<size_t> readDecomp3D)
{
    /* 1D decomposition */
    readerCount1D = shape1D / nproc;
    readerStart1D = rank * readerCount1D;
    int rem = shape1D % nproc;
    if (rank < rem)
    {
        ++readerCount1D;
        readerStart1D += rank;
    }
    else
    {
        readerStart1D += rem;
    }

    std::cout << "Reader rank " << rank << " 1D off = " << readerStart1D
              << ", cnt = " << readerCount1D << std::endl;

    /* 3D decomposition */
    /* Calculate 3D position of rank -> {rpx, rpy, rpz} */
    readerPos3D[0] = rank % readDecomp3D[0];
    size_t yz = rank / readDecomp3D[0];
    readerPos3D[1] = yz % readDecomp3D[1];
    size_t dxy = readDecomp3D[0] * readDecomp3D[1];
    readerPos3D[2] = rank / dxy;

    /* Calculate size of local array in 3D {ndx, ndy, ndz}
       and offset in global space in 3D {offx, offy, offz}
    */
    readerStart3D.resize(3);
    readerCount3D.resize(3);
    nElems3D = 1;
    for (int i = 0; i < 3; ++i)
    {
        readerCount3D[i] = shape3D[i] / readDecomp3D[i];
        readerStart3D[i] = readerPos3D[i] * readerCount3D[i];
        size_t rem = shape3D[i] % readDecomp3D[i];
        if (readerPos3D[i] < rem)
        {
            ++readerCount3D[i];
            readerStart3D[i] += readerPos3D[i];
        }
        else
        {
            readerStart3D[i] += rem;
        }
        nElems3D *= readerCount3D[i];
    }

    std::cout << "Reader rank " << rank << " pos = {" << readerPos3D[0] << ", "
              << readerPos3D[1] << ", " << readerPos3D[2] << "}"
              << " off = {" << readerStart3D[0] << ", " << readerStart3D[1]
              << ", " << readerStart3D[2] << "}"
              << " cnt = {" << readerCount3D[0] << ", " << readerCount3D[1]
              << ", " << readerCount3D[2] << "}" << std::endl;
}

Decomp::Decomp(const WarpxSettings &settings, MPI_Comm comm) : comm(comm)
{
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    std::string f1d = BroadcastFile(settings.inputfile1D, comm);
    ProcessDecomp1D(f1d);
    std::string f3d = BroadcastFile(settings.inputfile3D, comm);
    ProcessDecomp3D(f3d);
    DecomposeWriters();
    DecomposeReaders(settings.readDecomp3D);
}

Decomp::~Decomp()
{
    if (blocks1D)
    {
        free(blocks1D);
        blocks1D = nullptr;
    }
    if (blocks3D)
    {
        free(blocks3D);
        blocks3D = nullptr;
    }
}
