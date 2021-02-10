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

/* Computation and IO timers */
struct Timers
{
    Seconds compute;
    Seconds input;
    Seconds output;
    Timers() : compute(0.0), input(0.0), output(0.0){};
};

/* Timer functions */
std::vector<Timers> GatherTimers(const Timers &mytimers, MPI_Comm comm,
                                 int rank, int nproc);
Timers AvgTimes(std::vector<Timers> tv);

size_t MaxTimerIdx(std::vector<Timers> tv);

#endif /* TIMERS_H_ */
