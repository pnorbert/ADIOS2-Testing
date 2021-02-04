#include "decomp.h"

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <sstream>

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

void split(const std::string &s, char delimiter, std::vector<std::string>&v)
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

std::string Decomp::BroadcastFile(const std::string &fileName, MPI_Comm comm) const
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
        std::string fileContent = fileSS.str();
        length = fileContent.size();
        std::cout << "Length of " << fileName << " is " << length  << std::endl;
        std::cout << "Start: [" << fileContent.substr(0,20) << "]" << std::endl;
    }
    std::cout << "Length of " << fileName << " is " << fileContent.size()  << std::endl;
    std::cout << "Start: [" << fileContent.substr(0,20) << "]" << std::endl;

    if (nproc > 1)
    {
        MPI_Bcast(&length, 1, MPI_INT, 0, comm);
        if (rank > 0)
        {
            fileContent.resize(length);
        }
        MPI_Bcast(const_cast<char *>(fileContent.data()), length, MPI_CHAR, 0, comm);
    }
    return fileContent;
}

/* Process decomp_1D.in */
void Decomp::ReadDecomp1D(const std::string &fileContent) 
{
    std::string line;   
    std::cout << "Length of file is " << fileContent.size()  << std::endl;
    std::cout << "Start: [" << fileContent.substr(0,20) << "]" << std::endl;
    std::istringstream fin(fileContent);

    // Shape
    getline(fin,line);
    std::cout << "rank " << rank << " line 1: " << line << std::endl;
    std::vector<std::string> v(4);
    split(line, ' ', v);

    if (v[0] != "Shape")
    {
        throw std::runtime_error("First non-comment line of decomp_1D.in must be Shape ");
    }
    shape1D = std::stol(v[1]);
    
    // Blocks
    getline(fin,line);
    split(line, ' ', v);
    if (v[0] != "Blocks")
    {
        throw std::runtime_error("Second non-comment line of decomp_1D.in must be Blocks ");
    }
    nblocks1D = std::stol(v[1]);
    
    blocks1D = static_cast<Block1D*>(malloc(nblocks1D * sizeof(Block1D)));
    // Read blocks
    int blockID = 0;
    while(getline(fin,line)){
        if (!line.size() || line[0] == '#')
        {
            continue;
        }
        split(line, ' ', v);
        blocks1D[blockID].writerID=std::stol(v[1]);
        blocks1D[blockID].start=std::stol(v[2]);
        blocks1D[blockID].count=std::stol(v[3]);
        ++blockID;
    }

    std::cout << "decomp_1D.in: Shape = " << shape1D 
              << "  nBlocks = " << nblocks1D 
              << "  found = " << blockID
              << std::endl;

}

/* Process decomp_3D.in */
void Decomp::ReadDecomp3D(const std::string &fileContent)
{
    std::string line;  
    int maxWriterID = 0;
    std::istringstream fin(fileContent);

    // Shape
    getline(fin,line);
    std::vector<std::string> v(9);
    split(line, ' ', v);
    if (v[0] != "Shape")
    {
        throw std::runtime_error("First non-comment line of decomp_3D.in must be Shape ");
    }
    shape3D[0] = std::stol(v[1]);
    shape3D[1] = std::stol(v[2]);
    shape3D[2] = std::stol(v[3]);

    // Blocks
    getline(fin,line);
    split(line, ' ', v);
    if (v[0] != "Blocks")
    {
        throw std::runtime_error("Second non-comment line of decomp_3D.in must be Blocks ");
    }
    nblocks3D = std::stol(v[1]);
    
    blocks3D = static_cast<Block3D*>(malloc(nblocks3D * sizeof(Block3D)));
    // Read blocks
    int blockID = 0;
    while(getline(fin,line)){
        if (!line.size() || line[0] == '#')
        {
            continue;
        }
        split(line, ' ', v);
        blocks3D[blockID].writerID=std::stol(v[1]);
        if (blocks3D[blockID].writerID > maxWriterID)
        {
            maxWriterID = blocks3D[blockID].writerID;
        }
        blocks3D[blockID].start[0]=std::stol(v[2]);
        blocks3D[blockID].start[1]=std::stol(v[3]);
        blocks3D[blockID].start[2]=std::stol(v[4]);
        blocks3D[blockID].count[0]=std::stol(v[5]);
        blocks3D[blockID].count[1]=std::stol(v[6]);
        blocks3D[blockID].count[2]=std::stol(v[7]);
        ++blockID;
    }

    nWriters = maxWriterID + 1;
    std::cout << "decomp_3D.in: Shape = {" << shape3D[0]
              << " x " << shape3D[1] << " x " << shape3D[2] << "}"
              << "  nBlocks = " << nblocks3D
              << "  found = " << blockID
              << std::endl;
    std::cout << "nWriters = " << nWriters << std::endl;
}

void Decomp::DecomposeWriters()
{
    size_t ne = nWriters / nproc;
    size_t minWriterID = rank * ne;
    size_t rem = nWriters % nproc;
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
    std::cout << "rank " << rank << " writerIDs "
              << minWriterID << " - " << maxWriterID
              << std::endl;
}

Decomp::Decomp(MPI_Comm comm) : comm(comm)
{
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    std::string f1d = BroadcastFile("decomp_1D.in", comm);
    std::cout << "Length of file is " << f1d.size()  << std::endl;
    std::cout << "Start: [" << f1d.substr(0,20) << "]" << std::endl;
    ReadDecomp1D(f1d);
    std::string f3d = BroadcastFile("decomp_3D.in", comm);
    ReadDecomp3D(f3d);
    DecomposeWriters();
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
