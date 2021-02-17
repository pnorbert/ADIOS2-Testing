# Set the ADIOS2 installation directory
ADIOS2_DIR=/opt/adios2.debug

# Set the MPI compilers if available
MPICXX=mpicxx
CXXFLAGS=-g -std=c++11 -Wall

######################################################
#  These settings below should not need to be edited #
######################################################

CXX_MPI_FLAGS:=`${ADIOS2_DIR}/bin/adios2-config --cxx-flags -m`
CXX_MPI_LDFLAGS:=`${ADIOS2_DIR}/bin/adios2-config --cxx-libs -m`
 
.PHONY: clean all default
all: warpx-writer warpx-reader
default: all

warpx-writer: decomp.o warpxsettings.o timers.o io.o io_adios.o io_mpi.o io_posix.o warpx-writer.o
	${MPICXX} -o $@ ${CXXFLAGS} ${CXX_MPI_FLAGS} $+ ${CXX_MPI_LDFLAGS}

warpx-reader: decomp.o warpxsettings.o timers.o io.o io_adios.o io_mpi.o io_posix.o warpx-reader.o
	${MPICXX} -o $@ ${CXXFLAGS} ${CXX_MPI_FLAGS} $+ ${CXX_MPI_LDFLAGS}

%.o: %.cpp
	${MPICXX} -c ${CXXFLAGS} ${CXX_MPI_FLAGS} $+


clean: clean-data
	rm -rf warpx-writer warpx-reader
	rm -rf *.o 

clean-data:
	rm -rf *.bp *.sst *.ssc
	rm -rf warpx.[0-9]* warpx.idx
