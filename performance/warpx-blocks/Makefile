# Set the ADIOS2 installation directory
ADIOS2_DIR=/opt/adios2

# Set the MPI compilers if available
MPICXX=mpicxx
CXXFLAGS=-g -std=c++11

######################################################
#  These settings below should not need to be edited #
######################################################

CXX_MPI_FLAGS=`${ADIOS2_DIR}/bin/adios2-config --cxx-flags -m`
CXX_MPI_LDFLAGS=`${ADIOS2_DIR}/bin/adios2-config --cxx-libs -m`

all: warpx-writer
default: all

warpx-writer: decomp.o warpx-writer.o
	${MPICXX} -o warpx-writer ${CXXFLAGS} ${CXX_MPI_FLAGS} decomp.o warpx-writer.o ${CXX_MPI_LDFLAGS}

warpx-writer.o: decomp.o warpx-writer.cpp
	${MPICXX} -c ${CXXFLAGS} ${CXX_MPI_FLAGS} warpx-writer.cpp 

decomp.o: decomp.cpp 
	${MPICXX} -c ${CXXFLAGS} ${CXX_MPI_FLAGS} decomp.cpp 


clean:
	rm -rf warpx-writer
	rm -rf *.o 
	rm -rf *.bp *.sst *.ssc
