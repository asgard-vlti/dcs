#pragma once

#include "predictive_control.hpp"

#include <nlohmann/json.hpp>
#include <toml.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace heimdallr_ddspc {

inline nlohmann::json profile_json(const Parameters& p) {
    return {{"reg_start", p.reg_start},
            {"reg_cutoff", p.reg_cutoff},
            {"reg_divisor", p.reg_divisor},
            {"reg_interval", p.reg_interval},
            {"n_exploration", p.n_exploration},
            {"exploration_sigma", p.exploration_sigma},
            {"gamma", p.gamma}};
}

inline nlohmann::json status_json(const Parameters& configured,
                                  const Parameters* active) {
    return {{"configured", profile_json(configured)},
            {"active", active ? profile_json(*active) : nlohmann::json(nullptr)}};
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
    read_double("reg_start", p.reg_start);
    read_double("reg_cutoff", p.reg_cutoff);
    read_double("reg_divisor", p.reg_divisor);
    read_int("reg_interval", p.reg_interval);
    read_int("n_exploration", p.n_exploration);
    read_double("exploration_sigma", p.exploration_sigma);
    read_double("gamma", p.gamma);
    validate(p);
    return p;
}

inline void stage_parameter(Parameters& configured, const std::string& action,
                            const nlohmann::json& value) {
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
                                      const nlohmann::json& value) {
    if (action == "get") {
        if (!value.is_null()) {
            throw std::invalid_argument("ddspc get takes no value");
        }
    } else {
        stage_parameter(configured, action, value);
    }
    return status_json(configured, active);
}

}  // namespace heimdallr_ddspc
