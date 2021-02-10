#ifndef TIMERS_H_
#define TIMERS_H_

#include <chrono>
#include <vector>

#include <mpi.h>

typedef std::chrono::duration<double> Seconds;
typedef std::chrono::time_point<
    std::chrono::steady_clock,
    std::chrono::duration<double, std::chrono::steady_clock::period>>
    TimePoint;

/* Computation and IO timers
   total is about the sum of the others plus some unmeasured time
   io_mpi uses compute timer for the 1D and 3D block copy operations
   which is technically part of input
   io_mpi uses input for the time spent in receiving data with MPI
*/
struct Timers
{
    Seconds total;
    Seconds compute;
    Seconds input;
    Seconds output;
    Timers() : total(0.0), compute(0.0), input(0.0), output(0.0){};
};

/* Timer functions */
std::vector<Timers> GatherTimers(const Timers &mytimers, MPI_Comm comm,
                                 int rank, int nproc);
Timers AvgTimes(const std::vector<Timers> &tv);
size_t MaxTimerIdx(const std::vector<Timers> &tv);

#endif /* TIMERS_H_ */
