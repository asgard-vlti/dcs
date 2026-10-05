#pragma once

#include "predictive_control.hpp"

#include <atomic>
#include <cstdint>
#include <deque>

namespace heimdallr_ddspc {

struct ServoTransition {
    int from;
    int to;
    const char* trigger;
    std::uint64_t sequence;
    std::int64_t time_ns;
    Parameters ddspc_params;
};

inline void record_servo_transition(
    int& current_mode, int next_mode, const char* trigger,
    const Parameters& configured, std::int64_t time_ns,
    std::deque<ServoTransition>& transitions, std::uint64_t& next_sequence,
    std::atomic<std::uint64_t>& generation) {
    if (current_mode == next_mode) return;
    const std::uint64_t sequence = next_sequence + 1;
    transitions.push_back({current_mode, next_mode, trigger, sequence, time_ns,
                           configured});
    next_sequence = sequence;
    current_mode = next_mode;
    generation.fetch_add(1, std::memory_order_release);
}

}  // namespace heimdallr_ddspc
