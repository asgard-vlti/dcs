#pragma once

#include <cerrno>
#include <ctime>
#include <semaphore.h>
#include <system_error>

namespace heimdallr_ddspc {

inline bool wait_for_frame(sem_t* semaphore, bool& ready,
                           long timeout_nanoseconds = 10000000) {
    if (ready) return true;
    timespec deadline;
    if (clock_gettime(CLOCK_REALTIME, &deadline) != 0) {
        throw std::system_error(errno, std::generic_category(),
                                "FT clock_gettime failed");
    }
    deadline.tv_nsec += timeout_nanoseconds;
    deadline.tv_sec += deadline.tv_nsec / 1000000000;
    deadline.tv_nsec %= 1000000000;
    if (sem_timedwait(semaphore, &deadline) == 0) {
        ready = true;
        return true;
    }
    if (errno == EINTR || errno == ETIMEDOUT) return false;
    throw std::system_error(errno, std::generic_category(),
                            "FT frame wait failed");
}

}  // namespace heimdallr_ddspc
