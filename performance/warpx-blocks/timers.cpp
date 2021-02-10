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

Timers AvgTimes(const std::vector<Timers> &tv)
{
    Timers avg;
    for (const auto e : tv)
    {
        avg.total += e.total;
        avg.compute += e.compute;
        avg.input += e.input;
        avg.output += e.output;
    }
    avg.total = avg.total / tv.size();
    avg.compute = avg.compute / tv.size();
    avg.input = avg.input / tv.size();
    avg.output = avg.output / tv.size();
    return avg;
}

size_t MaxTimerIdx(const std::vector<Timers> &tv)
{
    size_t idx = 0;
    Seconds maxsum(0.0);
    for (size_t i = 0; i < tv.size(); ++i)
    {
        if (tv[i].total > maxsum)
        {
            idx = i;
            maxsum = tv[i].total;
        }
    }
    return idx;
}
