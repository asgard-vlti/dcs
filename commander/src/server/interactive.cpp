#include <commander/server/interactive.h>
#include <commander/server/inline_arguments.h>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <iostream>

namespace commander::server
{

    Interactive::Interactive(Module& module_):
        Interface(module_)
    {}

    void Interactive::run()
    {
        std::cout << "Interactive\n> ";

        std::string command;

        while(std::getline(std::cin, command))
        {
            if (command == "exit")
                break;

            auto pos = command.find(' ');
            std::string name;
            json args;

            try {


                if (pos == std::string::npos)
                    name = command;
                else
                {
                    name = command.substr(0, pos);
                    args = parse_inline_arguments(command.substr(pos + 1));
                }

                auto json = module_.execute(name, args);

                if (json.is_string())
                    fmt::print("{}\n> ", json.get<string>());
                else
                    fmt::print("{}\n> ", json.dump());

            } catch (const std::exception& e) {
                fmt::print("Error: {}\n> ", e.what());
            }
        }
    }


}
