#pragma once

#include <commander/type.h>

#include <boost/core/demangle.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <utility>

namespace commander
{

namespace detail
{

    template<typename T>
    string typename_of() {
        return boost::core::demangle(typeid(T).name());
    }

} // namespace detail

    struct UnamedArg
    {
        string type;
    };

    struct Arg : UnamedArg
    {
        string name;
        string description;
        optional<json> default_value;

        Arg() = default;

        Arg(string name):
            UnamedArg(),
            name(std::move(name)),
            description(),
            default_value()
        {}

        template<typename T>
        Arg& operator=(T&& value)
        {
            default_value = json(std::forward<T>(value));
            return *this;
        }

    };

    inline Arg arg(string name, string description)
    {
        Arg result(std::move(name));
        result.description = std::move(description);
        return result;
    }

    template<typename T>
    Arg arg(string name, string description, T&& default_value)
    {
        Arg result(std::move(name));
        result.description = std::move(description);
        result.default_value = json(std::forward<T>(default_value));
        return result;
    }

namespace literals
{

    inline Arg operator""_arg(const char* name, size_t size) {
        return Arg(string(name, size));
    }

} // namespace literals


    struct Command
    {
        string description;

        vector<Arg> arguments;
        UnamedArg return_value;
    };

} // namespace commander

template <> struct fmt::formatter<commander::UnamedArg> {

    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }

    template <typename FormatContext>
    constexpr auto format(const commander::UnamedArg& arg, FormatContext& ctx) const -> decltype(ctx.out()) {
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "{}", arg.type);
    }
};

template <> struct fmt::formatter<commander::Arg> {

    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    constexpr auto format(const commander::Arg& arg, FormatContext& ctx) const -> decltype(ctx.out()) {
        auto it = format_to(ctx.out(), "{}: {}", arg.name, arg.type);
        if (arg.default_value)
            it = format_to(it, " = {}", arg.default_value.value().dump());
        return it;
    }
};

template <> struct fmt::formatter<commander::Command> {

    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const commander::Command& cmd, FormatContext& ctx) -> decltype(ctx.out()) {
        // ctx.out() is an output iterator to write to.
        return format_to(ctx.out(), "({}) -> {} : {}", fmt::join(cmd.arguments, ", "), cmd.return_value, cmd.description);
    }
};
