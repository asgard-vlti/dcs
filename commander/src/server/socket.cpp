#include <commander/server/socket.h>
#include <commander/server/inline_arguments.h>

#include <fmt/core.h>
#include <iostream>

//#define DEBUG

namespace commander::server
{

    Socket::Socket(Module& module_, std::string socket):
        Interface(module_),
        ctx(1),
        // sock will listen for incoming messages
        sock(ctx, zmq::socket_type::rep)
    {
        fmt::print("[socket] binding to {}\n", socket);
        // sock will listen from socket
        sock.bind(socket);
    }


    void Socket::run()
    {

        zmq::message_t msg;
        while (true)
        {
            auto recv_res = sock.recv(msg, zmq::recv_flags::none);
            if (!recv_res)
            {
                fmt::print("[socket] error receiving message\n");
                continue;
            }

            auto command = std::string(static_cast<char*>(msg.data()), msg.size());
            
#ifdef DEBUG
            fmt::print("Received command: {}\n", command);
#endif

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

                // Treat the exit command as a special case
                if (name == "exit"){
                    fmt::print("Exiting socket server\n");
                    sock.send(zmq::message_t("Exiting!", 8), zmq::send_flags::none);
                    break;
                }
                auto res = module_.execute(name, args).dump();
#ifdef DEBUG
                fmt::print("Sending response: {}\n", res);
#endif

                sock.send(zmq::message_t(res.c_str(), res.size()), zmq::send_flags::none);
            } catch (const std::exception& e) {
                std::string error = fmt::format("Error: {}", e.what());
                fmt::print("{}\n", error);
                sock.send(zmq::message_t(error.c_str(), error.size()), zmq::send_flags::none);
            }
        }
    }
}
