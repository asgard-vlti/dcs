#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>

namespace commander::server {

inline nlohmann::json parse_inline_arguments(std::string_view source) {
    const std::string input(source);
    try {
        return nlohmann::json::parse("[" + input + "]");
    } catch (const nlohmann::json::parse_error&) {
        // Accept a quoted first argument followed by a space-separated value.
        const auto first = input.find_first_not_of(" \t");
        if (first == std::string::npos || input[first] != '"') throw;

        bool escaped = false;
        std::size_t end = first + 1;
        for (; end < input.size(); ++end) {
            if (escaped) {
                escaped = false;
            } else if (input[end] == '\\') {
                escaped = true;
            } else if (input[end] == '"') {
                break;
            }
        }
        if (end == input.size()) throw;

        const auto next = input.find_first_not_of(" \t", end + 1);
        if (next == std::string::npos || next == end + 1 ||
            input[next] == ',') {
            throw;
        }
        std::string with_comma = input;
        with_comma.insert(next, ", ");
        return nlohmann::json::parse("[" + with_comma + "]");
    }
}

}  // namespace commander::server
