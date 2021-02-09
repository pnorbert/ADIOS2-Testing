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
        blocks1D[blockID].producerID = std::stol(v[1]);
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
    size_t maxProducerID = 0;
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
        blocks3D[blockID].producerID = std::stol(v[1]);
        if (blocks3D[blockID].producerID > maxProducerID)
        {
            maxProducerID = blocks3D[blockID].producerID;
        }
        blocks3D[blockID].start[0] = std::stol(v[2]);
        blocks3D[blockID].start[1] = std::stol(v[3]);
        blocks3D[blockID].start[2] = std::stol(v[4]);
        blocks3D[blockID].count[0] = std::stol(v[5]);
        blocks3D[blockID].count[1] = std::stol(v[6]);
        blocks3D[blockID].count[2] = std::stol(v[7]);
        ++blockID;
    }

    nProducers = maxProducerID + 1;
    if (!rank)
    {
        std::cout << "decomp 3D: Shape = {" << shape3D[0] << " x " << shape3D[1]
                  << " x " << shape3D[2] << "}"
                  << "  nBlocks = " << nblocks3D << "  found = " << blockID
                  << std::endl;
        std::cout << "nProducers = " << nProducers << std::endl;
    }
}

void Decomp::DecomposeWriters()
{
    int b1d = 0;
    int b3d = 0;
    for (size_t r = 0; r < s.nWriters; ++r)
    {
        size_t ne = nProducers / s.nWriters;
        size_t minProducerID = r * ne;
        size_t rem = nProducers % s.nWriters;
        if (r < rem)
        {
            ++ne;
            minProducerID += r;
        }
        else
        {
            minProducerID += rem;
        }
        size_t maxProducerID = minProducerID + ne - 1;
        if (!rank && s.verbose >= 1)
        {
            std::cout << "Writer rank " << r << " producerIDs " << minProducerID
                      << " - " << maxProducerID << std::endl;
        }

        /* Record who is writing each 1D block */
        while (b1d < nblocks1D && blocks1D[b1d].producerID <= maxProducerID)
        {
            blocks1D[b1d].writerRank = r;
            ++b1d;
        }

        /* Record who is writing each 3D block */
        while (b3d < nblocks3D && blocks3D[b3d].producerID <= maxProducerID)
        {
            blocks3D[b3d].writerRank = r;
            ++b3d;
        }
    }
}

void Decomp::DecomposeReaders1D()
{
    for (size_t r = 0; r < s.nReaders; ++r)
    {
        /* 1D decomposition: each reader gets about the same number of blocks,
           not the same number of elements to ensure 1-to-1 sending of each
           block*/
        size_t nb = nblocks1D / s.nReaders;
        size_t sb = r * nb;
        size_t rem = nblocks1D % s.nReaders;
        if (r < rem)
        {
            ++nb;
            sb += r;
        }
        else
        {
            sb += rem;
        }

        readers[r].start1D = blocks1D[sb].start;
        readers[r].count1D = 0;
        for (size_t i = 0; i < nb; ++i)
        {
            readers[r].count1D += blocks1D[sb + i].count;
        }

        if (!rank && s.verbose >= 1)
        {
            std::cout << "Reader rank " << r << " 1D blocks " << sb << " - "
                      << sb + nb - 1 << " off = " << readers[r].start1D
                      << ", cnt = " << readers[r].count1D << std::endl;
        }

        /* Record who is reading each 1D block */
        for (size_t b = 0; b < nb; ++b)
        {
            blocks1D[sb + b].readerRank = static_cast<int>(r);
        }
    }
}

void Decomp::DecomposeReaders3D()
{
    for (size_t r = 0; r < s.nReaders; ++r)
    {
        /* 3D decomposition */
        /* Calculate 3D position of rank -> {rpx, rpy, rpz} */
        readers[r].pos3D[0] = r % s.readDecomp3D[0];
        size_t yz = r / s.readDecomp3D[0];
        readers[r].pos3D[1] = yz % s.readDecomp3D[1];
        size_t dxy = s.readDecomp3D[0] * s.readDecomp3D[1];
        readers[r].pos3D[2] = r / dxy;

        /* Calculate size of local array in 3D {ndx, ndy, ndz}
           and offset in global space in 3D {offx, offy, offz}
        */
        readers[r].nElems3D = 1;
        for (int i = 0; i < 3; ++i)
        {
            readers[r].count3D[i] = shape3D[i] / s.readDecomp3D[i];
            readers[r].start3D[i] = readers[r].pos3D[i] * readers[r].count3D[i];
            size_t rem = shape3D[i] % s.readDecomp3D[i];
            if (readers[r].pos3D[i] < rem)
            {
                ++readers[r].count3D[i];
                readers[r].start3D[i] += readers[r].pos3D[i];
            }
            else
            {
                readers[r].start3D[i] += rem;
            }
            readers[r].nElems3D *= readers[r].count3D[i];
        }

        if (!rank && s.verbose >= 2)
        {

            std::cout << "Reader rank " << r << " 3D pos = {"
                      << readers[r].pos3D[0] << ", " << readers[r].pos3D[1]
                      << ", " << readers[r].pos3D[2] << "}"
                      << " off = {" << readers[r].start3D[0] << ", "
                      << readers[r].start3D[1] << ", " << readers[r].start3D[2]
                      << "}"
                      << " cnt = {" << readers[r].count3D[0] << ", "
                      << readers[r].count3D[1] << ", " << readers[r].count3D[2]
                      << "}" << std::endl;
        }
    }
}

/* Record who is reading each 3D block */
void Decomp::CalculateReceivers3D()
{
    auto lf_FindReceiver = [&](const size_t posx, const size_t posy,
                               const size_t posz) -> int {
        size_t reader;
        for (size_t r = 0; r < s.nReaders; ++r)
        {
            if (readers[r].pos3D[0] == posx && readers[r].pos3D[1] == posy &&
                readers[r].pos3D[2] == posz)
            {
                reader = r;
                break;
            }
        }
        return reader;
    };

    size_t ndx = shape3D[0] / s.readDecomp3D[0];
    size_t ndy = shape3D[1] / s.readDecomp3D[1];
    size_t ndz = shape3D[2] / s.readDecomp3D[2];
    for (int b = 0; b < nblocks3D; ++b)
    {
        size_t posx = blocks3D[b].start[0] / ndx;
        size_t posy = blocks3D[b].start[1] / ndy;
        size_t posz = blocks3D[b].start[2] / ndz;
        /* Error checking */
        size_t posx2 = (blocks3D[b].start[0] + blocks3D[b].count[0] - 1) / ndx;
        size_t posy2 = (blocks3D[b].start[1] + blocks3D[b].count[1] - 1) / ndy;
        size_t posz2 = (blocks3D[b].start[2] + blocks3D[b].count[2] - 1) / ndz;
        if (posx != posx2 || posy != posy2 || posz != posz2)
        {
            if (!rank)
            {
                std::cout << "3D block " << b << " start = {"
                          << blocks3D[b].start[0] << ", "
                          << blocks3D[b].start[1] << ", "
                          << blocks3D[b].start[2] << "}"
                          << " would need to be split by multiple readers"
                          << ", which is not supported. Change the reader "
                          << "decomposition" << std::endl;
            }
            throw std::runtime_error(
                "Reader decomposition splits some writer's 3D block");
        }

        size_t reader = lf_FindReceiver(posx, posy, posz);
        blocks3D[b].readerRank = static_cast<int>(reader);
        /*if (!rank)
        {
            std::cout << "3D block " << b << " pos = {" << posx << ", " << posy
                      << ", " << posz << "}"
                      << " will be read by reader " << reader << std::endl;
        }*/
    }
}

Decomp::Decomp(const WarpxSettings &settings, MPI_Comm comm)
: s(settings), comm(comm)
{
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    std::string f1d = BroadcastFile(s.inputfile1D, comm);
    ProcessDecomp1D(f1d);
    std::string f3d = BroadcastFile(s.inputfile3D, comm);
    ProcessDecomp3D(f3d);

    DecomposeWriters();

    /* nReaders can be calculated on both sides from settings */
    readers.resize(s.nReaders);
    DecomposeReaders1D();
    DecomposeReaders3D();
    CalculateReceivers3D();
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
