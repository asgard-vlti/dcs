#pragma once

#include "predictive_control.hpp"
#include "servo_transitions.hpp"

#include <nlohmann/json.hpp>
#include <toml.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace heimdallr_ddspc {

inline nlohmann::json profile_json(const Parameters& p) {
    return {{"history", p.history},
            {"future", p.future},
            {"reg_start", p.reg_start},
            {"reg_cutoff", p.reg_cutoff},
            {"reg_divisor", p.reg_divisor},
            {"reg_interval", p.reg_interval},
            {"n_exploration", p.n_exploration},
            {"exploration_sigma", p.exploration_sigma},
            {"gamma", p.gamma},
            {"continue_learning", p.continue_learning}};
}

inline nlohmann::json status_json(const Parameters& configured,
                                  const Parameters* active,
                                  const FreezeStatus* freeze_status = nullptr) {
    nlohmann::json runtime = nullptr;
    if (active && freeze_status) {
        runtime = {{"freeze_pending", freeze_status->pending},
                   {"frozen", freeze_status->frozen},
                   {"freeze_reason", freeze_status->reason
                                         ? nlohmann::json(freeze_status->reason)
                                         : nlohmann::json(nullptr)}};
    }
    return {{"configured", profile_json(configured)},
            {"active", active ? profile_json(*active) : nlohmann::json(nullptr)},
            {"runtime", std::move(runtime)}};
}

inline Parameters read_config(const toml::table& table) {
    Parameters p;
    const auto* section = table["ddspc"].as_table();
    if (!section) {
        if (table.contains("ddspc")) {
            throw std::invalid_argument("[ddspc] must be a TOML table");
        }
        return p;
    }
    auto read_double = [&](const char* key, double& field) {
        const auto node = (*section)[key];
        if (!node) return;
        const auto parsed = node.value<double>();
        if (!parsed) {
            throw std::invalid_argument(std::string("ddspc.") + key +
                                        " must be a number");
        }
        field = *parsed;
    };
    auto read_int = [&](const char* key, int& field) {
        const auto node = (*section)[key];
        if (!node) return;
        const auto parsed = node.value<int64_t>();
        if (!parsed || *parsed < std::numeric_limits<int>::min() ||
            *parsed > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(std::string("ddspc.") + key +
                                        " must be an integer in range");
        }
        field = static_cast<int>(*parsed);
    };
    auto read_bool = [&](const char* key, bool& field) {
        const auto node = (*section)[key];
        if (!node) return;
        const auto* parsed = node.as_boolean();
        if (!parsed) {
            throw std::invalid_argument(std::string("ddspc.") + key +
                                        " must be a boolean");
        }
        field = parsed->get();
    };
    read_double("reg_start", p.reg_start);
    read_double("reg_cutoff", p.reg_cutoff);
    read_double("reg_divisor", p.reg_divisor);
    read_int("reg_interval", p.reg_interval);
    read_int("n_exploration", p.n_exploration);
    read_double("exploration_sigma", p.exploration_sigma);
    read_double("gamma", p.gamma);
    read_bool("continue_learning", p.continue_learning);
    validate(p);
    return p;
}

inline void stage_parameter(Parameters& configured, const std::string& action,
                            const nlohmann::json& value) {
    if (action == "set-continue-learning") {
        if (!value.is_boolean()) {
            throw std::invalid_argument("ddspc continue_learning must be a boolean");
        }
        configured.continue_learning = value.get<bool>();
        return;
    }
    if (!value.is_number()) {
        throw std::invalid_argument("ddspc setter requires a numeric value");
    }
    Parameters next = configured;
    if (action == "set-reg-interval" || action == "set-n-exploration") {
        if (!value.is_number_integer()) {
            throw std::invalid_argument("ddspc frame count must be an integer");
        }
        int count;
        if (value.is_number_unsigned()) {
            const uint64_t parsed = value.get<uint64_t>();
            if (parsed > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("ddspc frame count out of range");
            }
            count = static_cast<int>(parsed);
        } else {
            const int64_t parsed = value.get<int64_t>();
            if (parsed < std::numeric_limits<int>::min() ||
                parsed > std::numeric_limits<int>::max()) {
                throw std::invalid_argument("ddspc frame count out of range");
            }
            count = static_cast<int>(parsed);
        }
        if (action == "set-reg-interval") next.reg_interval = count;
        else next.n_exploration = count;
    } else {
        const double number = value.get<double>();
        if (action == "set-reg-start") next.reg_start = number;
        else if (action == "set-reg-cutoff") next.reg_cutoff = number;
        else if (action == "set-reg-divisor") next.reg_divisor = number;
        else if (action == "set-exploration-sigma")
            next.exploration_sigma = number;
        else if (action == "set-gamma") next.gamma = number;
        else throw std::invalid_argument("Unknown ddspc action: " + action);
    }
    validate(next);
    configured = next;
}

inline nlohmann::json execute_command(Parameters& configured,
                                      const Parameters* active,
                                      const std::string& action,
                                      const nlohmann::json& value,
                                      const FreezeStatus* freeze_status = nullptr,
                                      bool servo_off = false) {
    if (action == "get") {
        if (!value.is_null()) {
            throw std::invalid_argument("ddspc get takes no value");
        }
    } else if (action == "set-history") {
        if (!servo_off) {
            throw std::invalid_argument("ddspc history can change only with servo off");
        }
        if (!value.is_array() || value.size() != 2 ||
            !value[0].is_number_integer() || !value[1].is_number_integer()) {
            throw std::invalid_argument("ddspc history requires [history,future] integers");
        }
        const auto history = value[0].get<int64_t>();
        const auto future = value[1].get<int64_t>();
        if (history < std::numeric_limits<int>::min() ||
            history > std::numeric_limits<int>::max() ||
            future < std::numeric_limits<int>::min() ||
            future > std::numeric_limits<int>::max() ||
            !supported_history_preset(static_cast<int>(history),
                                      static_cast<int>(future))) {
            throw std::invalid_argument("Unsupported ddspc history preset");
        }
        configured.history = static_cast<int>(history);
        configured.future = static_cast<int>(future);
    } else {
        stage_parameter(configured, action, value);
    }
    return status_json(configured, active, freeze_status);
}

inline void queue_freeze_request(
    int current_mode, int ddspc_mode, const Parameters* active,
    const nlohmann::json& value, FreezeStatus& status, std::int64_t time_ns,
    std::deque<ServoTransition>& transitions, std::uint64_t& next_sequence,
    std::atomic<std::uint64_t>& generation) {
    if (!value.is_null()) {
        throw std::invalid_argument("ddspc freeze takes no value");
    }
    if (!active || current_mode != ddspc_mode) {
        throw std::invalid_argument("DDSPC mode is not active");
    }
    if (status.pending || status.frozen) return;
    record_freeze_request(current_mode, ddspc_mode, *active, time_ns,
                          transitions, next_sequence, generation);
    status.pending = true;
}

}  // namespace heimdallr_ddspc
