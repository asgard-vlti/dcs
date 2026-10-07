#define TOML_HEADER_ONLY 0
#define TOML_IMPLEMENTATION
#include "../ddspc_tuning.hpp"

#include <commander/commander.h>
#include <commander/server/inline_arguments.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

using heimdallr_ddspc::Modes;
using heimdallr_ddspc::Parameters;

int rejection_count = 0;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Fn>
void rejects(Fn fn) {
    ++rejection_count;
    try {
        fn();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Expected invalid value to be rejected: " +
                             std::to_string(rejection_count));
}

int main() {
    using namespace heimdallr_ddspc;

    Parameters configured = read_config(toml::parse("name = 'defaults'"));
    check(configured.reg_start == 1e5 && configured.reg_cutoff == 0.2 &&
              configured.reg_divisor == 5.0 && configured.reg_interval == 100 &&
              configured.n_exploration == 500 &&
              configured.exploration_sigma == 0.01 && configured.gamma == 1.0 &&
              configured.continue_learning,
          "Incorrect default profile");

    const auto overrides = toml::parse(
        "[ddspc]\nreg_start = 100.0\nreg_cutoff = 0.2\n"
        "reg_divisor = 10.0\nreg_interval = 2\nn_exploration = 2\n"
        "exploration_sigma = 0.1\ngamma = 0.5\ncontinue_learning = false\n");
    configured = read_config(overrides);
    check(configured.reg_start == 100.0 && configured.reg_divisor == 10.0 &&
              configured.reg_interval == 2 && configured.n_exploration == 2 &&
              configured.exploration_sigma == 0.1 && configured.gamma == 0.5 &&
              !configured.continue_learning,
          "Incorrect TOML overrides");
    rejects([&] { read_config(toml::parse("ddspc = 1")); });
    rejects([&] { read_config(toml::parse("[ddspc]\ngamma = 0.0")); });
    rejects([&] {
        read_config(toml::parse("[ddspc]\nreg_interval = 'bad'"));
    });
    rejects([&] {
        read_config(toml::parse("[ddspc]\ncontinue_learning = 1"));
    });
    check(read_config(toml::parse("[ddspc]\nreg_start = 100000"))
                  .reg_start == 1e5,
          "Integer TOML value was not accepted for a numeric parameter");
    check(read_config(toml::parse_file("def.toml")).reg_cutoff == 0.3,
          "Packaged default TOML is invalid");

    Parameters active = configured;
    const auto before = execute_command(configured, &active, "get", nullptr);
    check(before["configured"] == before["active"], "Initial profiles differ");
    const auto after = execute_command(configured, &active, "set-reg-start", 200.0);
    check(after["configured"]["reg_start"] == 200.0 &&
              after["active"]["reg_start"] == 100.0,
          "Setter changed the active profile");
    const auto staged_freeze = execute_command(
        configured, &active, "set-continue-learning", true);
    check(staged_freeze["configured"]["continue_learning"] == true &&
              staged_freeze["active"]["continue_learning"] == false,
          "Boolean setter changed the active profile");
    rejects([&] {
        execute_command(configured, &active, "set-continue-learning", 1);
    });
    rejects([&] {
        execute_command(configured, &active, "set-continue-learning", "false");
    });
    rejects([&] { execute_command(configured, &active, "set-reg-cutoff", 300.0); });
    rejects([&] { execute_command(configured, &active, "set-gamma", 1.1); });
    rejects([&] { execute_command(configured, &active, "set-reg-interval", 2.5); });
    rejects([&] { execute_command(configured, &active, "set-n-exploration", -1); });
    rejects([&] { execute_command(configured, &active, "set-reg-start", nullptr); });
    rejects([&] { execute_command(configured, &active, "get", 1); });
    rejects([&] { execute_command(configured, &active, "unknown", 1); });
    check(configured.reg_start == 200.0 && configured.reg_cutoff == 0.2 &&
              configured.gamma == 0.5 && configured.reg_interval == 2,
          "Invalid setter changed configured profile");
    check(execute_command(configured, nullptr, "get", nullptr)["active"].is_null(),
          "Inactive profile should be null");
    active = configured;
    check(execute_command(configured, &active, "get", nullptr)["active"]
                  ["reg_start"] == 200.0,
          "Mode entry did not adopt staged profile");

    FreezeStatus freeze_status;
    std::deque<ServoTransition> freeze_events;
    std::uint64_t freeze_sequence = 0;
    std::atomic<std::uint64_t> freeze_generation{0};
    rejects([&] {
        queue_freeze_request(4, 5, nullptr, nullptr, freeze_status, 1,
                             freeze_events, freeze_sequence, freeze_generation);
    });
    rejects([&] {
        queue_freeze_request(5, 5, &active, true, freeze_status, 1,
                             freeze_events, freeze_sequence, freeze_generation);
    });
    queue_freeze_request(5, 5, &active, nullptr, freeze_status, 2,
                         freeze_events, freeze_sequence, freeze_generation);
    queue_freeze_request(5, 5, &active, nullptr, freeze_status, 3,
                         freeze_events, freeze_sequence, freeze_generation);
    check(freeze_status.pending && freeze_events.size() == 1 &&
              freeze_generation.load() == 1 &&
              status_json(configured, &active, &freeze_status)["runtime"]
                  ["freeze_pending"] == true,
          "Repeated freeze queued more than one request");
    freeze_status = {false, true, "manual"};
    queue_freeze_request(5, 5, &active, nullptr, freeze_status, 4,
                         freeze_events, freeze_sequence, freeze_generation);
    check(freeze_events.size() == 1 &&
              status_json(configured, &active, &freeze_status)["runtime"]
                  ["freeze_reason"] == "manual",
          "Frozen run did not report its state or remain idempotent");

    commander::Module commands;
    commands.def(
        "ddspc",
        [&](std::string action, nlohmann::json value) {
            return execute_command(configured, &active, action, value);
        },
        "DDSPC tuning", commander::arg("action", "Subcommand"),
        commander::arg("value", "Value", nlohmann::json(nullptr)));
    const std::string wire = "ddspc \"set-reg-start\" 300.0";
    const auto split = wire.find(' ');
    const auto args = commander::server::parse_inline_arguments(
        wire.substr(split + 1));
    const auto reply = commands.execute(wire.substr(0, split), args);
    check(reply["configured"]["reg_start"] == 300.0 &&
              reply["active"]["reg_start"] == 200.0,
          "Commander did not parse the staged setter");
    const auto comma_args = commander::server::parse_inline_arguments(
        "\"set-reg-start\", 400.0");
    check(comma_args == nlohmann::json::array({"set-reg-start", 400.0}),
          "Comma-separated syntax changed");
    const auto interval_args = commander::server::parse_inline_arguments(
        "\"set-reg-interval\" 200");
    const auto interval_reply = commands.execute("ddspc", interval_args);
    check(interval_reply["configured"]["reg_interval"] == 200 &&
              interval_reply["active"]["reg_interval"] == 2,
          "Space-separated reg interval command failed");
    const auto bool_args = commander::server::parse_inline_arguments(
        "\"set-continue-learning\" false");
    const auto bool_reply = commands.execute("ddspc", bool_args);
    check(bool_reply["configured"]["continue_learning"] == false &&
              bool_reply["active"]["continue_learning"] == true,
          "Commander did not parse the boolean setter");
    const auto freeze_args = commander::server::parse_inline_arguments(
        "\"freeze\"");
    check(freeze_args == nlohmann::json::array({"freeze"}),
          "Commander did not parse the no-value freeze command");
    commands.execute("ddspc", nlohmann::json::array({"set-reg-interval", 2}));
    const auto get_reply = commands.execute("ddspc", nlohmann::json::array({"get"}));
    check(get_reply["configured"]["reg_start"] == 300.0,
          "Commander did not apply the optional argument default");
    const auto bad_reply = commands.execute(
        "ddspc", nlohmann::json::array({"set-reg-interval", 1.5}));
    check(bad_reply.contains("error") && configured.reg_interval == 2,
          "Commander accepted an invalid integer setter");

    Parameters control_params = configured;
    control_params.reg_start = 100.0;
    PredictiveControl<4, 2> control(control_params);
    control.advance_regularization(0);
    check(control.regularization() == 10.0, "Initial division missing");
    control.advance_regularization(1);
    check(control.regularization() == 10.0, "Division interval ignored");
    control.advance_regularization(2);
    check(control.regularization() == 1.0, "Second division missing");
    control.advance_regularization(4);
    check(control.regularization() == 0.2, "Hard floor ignored");
    control.advance_regularization(6);
    check(control.regularization() == 0.2, "Regularization fell below floor");

    const Modes zero = Modes::Zero();
    const Modes draw = Modes::Constant(10.0);
    check((control.propose(zero, draw) - Modes::Constant(0.4)).norm() < 1e-12,
          "Exploration sigma or clamp ignored");
    control.update(control.command());
    check((control.propose(zero, draw) - Modes::Constant(0.8)).norm() < 1e-12,
          "Second exploration frame missing");
    control.update(control.command());
    check((control.propose(zero, draw) - Modes::Constant(0.8)).norm() < 1e-12,
          "Exploration exceeded configured duration");

    control.update(control.command(), false);
    check(control.rls_updates() == 0 && control.iterations() == 3,
          "Frozen controller did not keep frame history without learning");

    QrdRls<1, 1> rls(1.0, control_params.gamma);
    Eigen::Matrix<double, 1, 1> feature = Eigen::Matrix<double, 1, 1>::Zero();
    rls.update(feature, feature);
    check(std::abs(rls.gram(0, 0) - 0.5) < 1e-12,
          "Gamma did not scale QRD RLS history");

    std::cout << "DDSPC parameter checks passed\n";
    return 0;
}
