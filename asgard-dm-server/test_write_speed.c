#include <stdio.h>
#include <time.h>

#define DURATION_NS 100000000LL  /* 0.1 seconds in nanoseconds */

static long long timespec_to_ns(struct timespec *ts) {
    return (long long)ts->tv_sec * 1000000000LL + ts->tv_nsec;
}

int main(void) {
    FILE *f1 = fopen("timestamps1.txt", "w");
    FILE *f2 = fopen("timestamps2.txt", "w");
    if (!f1 || !f2) {
        perror("fopen");
        return 1;
    }

    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    long long t_start = timespec_to_ns(&ts_start);
    long long count = 0;

    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &ts_now);
        long long t_now = timespec_to_ns(&ts_now);
        if (t_now - t_start >= DURATION_NS)
            break;

        fprintf(f1, "%lld.%09ld\n", (long long)ts_now.tv_sec, ts_now.tv_nsec);
        fprintf(f2, "%lld.%09ld\n", (long long)ts_now.tv_sec, ts_now.tv_nsec);
        count++;
    }

    fclose(f1);
    fclose(f2);

    struct timespec ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (timespec_to_ns(&ts_end) - t_start) / 1e9;
    printf("Wrote %lld lines to each file in %.4f s (%.0f lines/s)\n",
           count, elapsed, count / elapsed);

    return 0;
}
