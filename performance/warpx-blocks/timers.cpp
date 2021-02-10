#include "timers.h"

std::vector<Timers> GatherTimers(const Timers &mytimers, MPI_Comm comm,
                                 int rank, int nproc)
{
    std::vector<Timers> v;
    if (!rank)
        v.resize(nproc);
    MPI_Gather(&mytimers, sizeof(Timers), MPI_BYTE, v.data(), sizeof(Timers),
               MPI_BYTE, 0, comm);
    return v;
}

double AvgTime(std::vector<double> v)
{
    double sum = 0;
    for (const auto e : v)
        sum += e;
    return sum / v.size();
}

Timers AvgTimes(std::vector<Timers> tv)
{
    Timers avg;
    for (const auto e : tv)
    {
        avg.compute += e.compute;
        avg.input += e.input;
        avg.output += e.output;
    }
    avg.compute = avg.compute / tv.size();
    avg.input = avg.input / tv.size();
    avg.output = avg.output / tv.size();
    return avg;
}

size_t MaxTimerIdx(std::vector<Timers> tv)
{
    Timers avg;
    size_t idx = 0;
    Seconds maxsum(0.0);
    for (size_t i = 0; i < tv.size(); ++i)
    {
        Seconds sum = tv[i].compute + tv[i].input + tv[i].output;
        if (sum > maxsum)
        {
            idx = i;
            maxsum = sum;
        }
    }
    return idx;
}