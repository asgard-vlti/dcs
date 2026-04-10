#include <commander/commander.h>

#include <cerrno>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

static int single_instance_lock_fd = -1;

bool acquire_single_instance_lock(const char *lock_path) {
    single_instance_lock_fd = open(lock_path, O_RDWR | O_CREAT, 0644);
    if (single_instance_lock_fd < 0) {
        logprintf(LOG_ERROR, "Failed to open lock file %s: %s. '", lock_path, std::strerror(errno));
        return false;
    }

    if (flock(single_instance_lock_fd, LOCK_EX | LOCK_NB) != 0) {
        if (errno == EWOULDBLOCK) {
            logprintf(LOG_ERROR, "Another heimdallr server is already running (lock file: %s). ", lock_path);
        } else {
            logprintf(LOG_ERROR, "Failed to lock %s: %s", lock_path, std::strerror(errno));
        }
        close(single_instance_lock_fd);
        single_instance_lock_fd = -1;
        return false;
    }

    // Best-effort PID write for operator visibility.
    if (ftruncate(single_instance_lock_fd, 0) == 0) {
        char pid_buf[32];
        int n = std::snprintf(pid_buf, sizeof(pid_buf), "%ld\n", (long)getpid());
        if (n > 0) {
            if (write(single_instance_lock_fd, pid_buf, (size_t)n) < 0) {
                logprintf(LOG_WARNING, "Failed to write PID to lock file %s: %s", lock_path, std::strerror(errno));
            }
        }
    }

    return true;
}

/* =========================================================================
 *         Like printf, but prepends an ISO 8601 UTC timestamp
 * ========================================================================= */
void logprintf(int loglevel, const char *fmt, ...) {
  // Lower loglevels are more important. 
  if (loglevel > settings.s.loglevel) {
    return;
  }
  std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  struct tm *tm_info = gmtime(&now);
  char timebuf[21];
  strftime(timebuf, sizeof(timebuf), "%Y-%m-%dT%H:%M:%SZ", tm_info);
  printf("%s ", timebuf);
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);
  printf("\n");
}