#include <commander/server.h>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/ostream.h>

#include <boost/program_options.hpp>

#include <variant>
#include <vector>
#include <string>

#include <commander/server/interactive.h>
#include <commander/server/single_command.h>
#include <commander/server/socket.h>


namespace po = boost::program_options;

namespace commander
{
    
    // std::tuple<std::string, nlohmann::json> parse_command(std::string cmd)
    // {
    //     std::string command;
    //     nlohmann::json params;

    //     auto pos = cmd.find(' ');
    //     if (pos != std::string::npos)
    //     {
    //         command = cmd.substr(0, pos);
    //         params = nlohmann::json::parse(cmd.substr(pos + 1));
    //     }
    //     else
    //     {
    //         command = cmd;
    //     }

    //     return std::make_tuple(command, params);
    // }



    struct Interactive
    {
        static po::options_description options() {
            return {"Interactive"};
        }
    };

    struct Socket
    {
        static po::options_description options() {
            po::options_description config{"Socket"};
            config.add_options()
                ("socket", po::value<std::string>(), "Socket path")
                ;

            return config;
        }

        std::string socket;
    };

    struct SingleCommand
    {
        static po::options_description options() {
            po::options_description config{"SingleCommand"};
            config.add_options()
                ("command", po::value<std::vector<std::string>>()->multitoken(), "Command to execute")
                ;

            return config;
        }

        std::string command;
    };

    using Mode = std::variant<Interactive, Socket, SingleCommand>;


    Mode parse_options(int argc, char** argv)
    {
        po::options_description cmdline_options;
        cmdline_options.add_options()
            ("help", "Print help message")
            // ("interactive", "Interactive mode")
            // ("socket", po::value<std::string>(), "Socket path")
            // ("command", po::value<std::string>(), "Command to execute")
            ;

        cmdline_options.add(Interactive::options()).add(Socket::options()).add(SingleCommand::options());
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
        po::notify(vm);

        if (vm.count("help")) {
            fmt::print("{}\n", cmdline_options);
            std::exit(0);
        }

        if (vm.count("socket"))
            return Socket{vm["socket"].as<std::string>()};

        if (vm.count("command"))
        {
            auto args = vm["command"].as<std::vector<std::string>>();
            return SingleCommand{fmt::format("{}", fmt::join(args, " "))};
        }

        // By default, interactive mode.
        return Interactive{};
    }

    Server::Server(int argc, char** argv):
        module_()
    {
        auto cmd = parse_options(argc, argv);

        this->impl = std::visit(
            [&](auto&& arg) -> std::shared_ptr<server::Interface>
            {
                using T = std::decay_t<decltype(arg)>;

                if constexpr (std::is_same_v<T, Interactive>)
                {
                    // Interactive mode
                    return std::make_shared<server::Interactive>(module_);
                }
                else if constexpr (std::is_same_v<T, Socket>)
                {
                    // Socket mode
                    return std::make_shared<server::Socket>(module_, arg.socket);
                }
                else if constexpr (std::is_same_v<T, SingleCommand>)
                {
                    // Single command mode
                    return std::make_shared<server::SingleCommand>(module_, arg.command);
                }
            },
            cmd
        );
    }

    void Server::run()
    {
        this->impl->run();
    }

}
