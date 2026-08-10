#pragma once

#include <commander/module.h>
#include <commander/function_parser.h>
#include <commander/registry.h>
#include <commander/server.h>

// For logging
#define LOG_ERROR 1
#define LOG_WARNING 2
#define LOG_INFO 3
#define LOG_DEBUG 4
#define LOG_LEVEL_NAMES {"ERROR", "WARNING", "INFO", "DEBUG"}
#define LOG_LEVEL_COLORS {"\033[31m", "\033[33m", "\033[0m", "\033[34m"} // Red, Yellow, Default, Blue

void logprintf(int loglevel, const char *fmt, ...);
bool acquire_single_instance_lock(const char *lock_path);
void unacquire_single_instance_lock();
int set_log_level(int level);

// Shortcuts for common log levels (using heimdallr's settings.s.loglevel)
#define error(...) logprintf(LOG_ERROR, __VA_ARGS__)
#define warn(...) logprintf(LOG_WARNING, __VA_ARGS__)
#define info(...) logprintf(LOG_INFO, __VA_ARGS__)
#define debug(...) logprintf(LOG_DEBUG, __VA_ARGS__)
