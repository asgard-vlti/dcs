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

void logprintf(int loglevel, const char *fmt, ...);
bool acquire_single_instance_lock(const char *lock_path);
