#include <commander/module.h>

namespace commander
{
        Module::Module():
            functions(),
            commands()
        {
            register_functions(*this);

            def("help", [this]() {
                return get_help();
            }, "List every command and its description.");
            def("command_names", [this]() {
                return command_names();
            }, "List all available command names.");
            def("description", [this](string name) {
                return description(name);
            }, "Describe one command, including its signature.",
                arg("name", "Name of the command to describe."));
            def("signature", [this](string name) {
                return signature(name);
            }, "Return the arguments and return type for one command.",
                arg("name", "Name of the command to inspect."));
            def("arguments", [this](string name) {
                return arguments(name);
            }, "Return argument metadata for one command.",
                arg("name", "Name of the command to inspect."));
            def("return_type", [this](string name) {
                return return_type(name);
            }, "Return the result type for one command.",
                arg("name", "Name of the command to inspect."));
        }

        string Module::get_help() const {
            string help;
            for (auto& [name, cmd]: commands)
                help = fmt::format("{}{}: {}\n", help, name, cmd.description);
            return help;
        }

        std::vector<string> Module::command_names() const {
            std::vector<string> names; names.reserve(commands.size());
            for(auto& e : commands | boost::adaptors::map_keys)
                names.push_back(e);
            return names;
        }

        string Module::description(const string& name) const {
            return fmt::to_string(commands.at(name));
        }

        json Module::signature(const string& name) const {
            json res;
            res["arguments"] = arguments(name);
            res["return_type"] = return_type(name);
            return res;
        }

        json Module::arguments(const string& name) const {
            auto fn = commands.at(name);
            json res;
            for(auto& arg : fn.arguments) {
                json arg_json;
                arg_json["name"] = arg.name;
                arg_json["type"] = arg.type;
                if (!arg.description.empty())
                    arg_json["description"] = arg.description;
                if (arg.default_value)
                    arg_json["default_value"] = arg.default_value.value();
                res.push_back(arg_json);
            }
            return res;
        }

        json Module::return_type(const string& name) const {
            return commands.at(name).return_value.type;
        }

        json Module::execute(const string& name, const json& args)
        {
            try {
                return functions.at(name)(args);
            } catch (const std::exception& e) {
                return json{{"error", e.what()}};
            }
        }

} // namespace commander
