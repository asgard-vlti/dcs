#include "../predictive_control.hpp"

#include <iomanip>
#include <iostream>
#include <string>

using heimdallr_ddspc::Modes;
using heimdallr_ddspc::PredictiveControl;
using heimdallr_ddspc::Telescopes;

template <int HistoryLength, int FutureLength>
int run_controller(int count) {
    PredictiveControl<HistoryLength, FutureLength> controller;
    for (int i = 0; i < count; ++i) {
        Modes error, draw, applied;
        for (int j = 0; j < 3; ++j) std::cin >> error(j);
        for (int j = 0; j < 3; ++j) std::cin >> draw(j);
        for (int j = 0; j < 3; ++j) std::cin >> applied(j);
        if (!std::cin) return 2;
        controller.advance_regularization(i);
        const Modes proposed = controller.propose(error, draw);
        controller.update(applied);
        std::cout << proposed.transpose() << '\n';
    }

    constexpr int features = PredictiveControl<HistoryLength, FutureLength>::Features;
    constexpr int outputs = PredictiveControl<HistoryLength, FutureLength>::Outputs;
    constexpr int control_features =
        PredictiveControl<HistoryLength, FutureLength>::ControlFeatures;
    std::cout << "REG " << controller.regularization() << '\n';
    std::cout << "WEIGHTS";
    for (int i = 0; i < features; ++i) {
        for (int j = 0; j < outputs; ++j) {
            std::cout << ' ' << controller.rls().weights()(i, j);
        }
    }
    std::cout << '\n';
    std::cout << "GRAM";
    for (int i = 0; i < features; ++i) {
        for (int j = 0; j < features; ++j) {
            std::cout << ' ' << controller.rls().gram(i, j);
        }
    }
    std::cout << '\n';
    std::cout << "PREDICTIVE";
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < control_features; ++j) {
            std::cout << ' ' << controller.predictive()(i, j);
        }
    }
    std::cout << '\n';
    return 0;
}

int main() {
    std::cout << std::setprecision(17);
    std::string mode;
    std::cin >> mode;
    if (mode == "transform") {
        double wavelength;
        Telescopes phase, dm;
        std::cin >> wavelength;
        for (int i = 0; i < 4; ++i) std::cin >> phase(i);
        for (int i = 0; i < 4; ++i) std::cin >> dm(i);
        const Modes error = heimdallr_ddspc::phase_error_modes(phase);
        const Telescopes command = heimdallr_ddspc::dm_command(error, wavelength);
        const Modes feedback =
            heimdallr_ddspc::applied_command_waves(dm, wavelength);
        std::cout << error.transpose() << '\n';
        std::cout << command.transpose() << '\n';
        std::cout << feedback.transpose() << '\n';
        return 0;
    }
    if (mode == "servo") {
        double wavelength, opd_per_dm_unit, dm_limit;
        Telescopes current_dm;
        std::cin >> wavelength >> opd_per_dm_unit >> dm_limit;
        for (int i = 0; i < 4; ++i) std::cin >> current_dm(i);
        heimdallr_ddspc::DdspcServo servo;
        servo.enter();
        const Telescopes phase = Telescopes::Zero();
        const Modes zero = Modes::Zero();
        const Modes large_draw = Modes::Constant(100.0);
        const Telescopes first = servo.propose(
            phase, current_dm, wavelength, opd_per_dm_unit, dm_limit, zero);
        servo.update(first, wavelength, opd_per_dm_unit);
        current_dm = first;
        for (int i = 1; i < 500; ++i) {
            current_dm = servo.propose(phase, current_dm, wavelength,
                                       opd_per_dm_unit, dm_limit, zero);
            servo.update(current_dm, wavelength, opd_per_dm_unit);
        }
        servo.invalidate();
        const Telescopes after_loss = servo.propose(
            phase, current_dm, wavelength, opd_per_dm_unit, dm_limit,
            large_draw);
        servo.enter();
        const Telescopes after_entry = servo.propose(
            phase, current_dm, wavelength, opd_per_dm_unit, dm_limit,
            large_draw);
        std::cout << first.transpose() << '\n';
        std::cout << after_loss.transpose() << '\n';
        std::cout << after_entry.transpose() << '\n';
        return 0;
    }
    if (mode == "servo_trace") {
        int count;
        double wavelength, opd_per_dm_unit, dm_limit;
        Telescopes current_dm;
        std::cin >> count >> wavelength >> opd_per_dm_unit >> dm_limit;
        for (int i = 0; i < 4; ++i) std::cin >> current_dm(i);
        heimdallr_ddspc::DdspcServo servo;
        servo.enter();
        for (int i = 0; i < count; ++i) {
            Telescopes phase;
            Modes draw;
            for (int j = 0; j < 4; ++j) std::cin >> phase(j);
            for (int j = 0; j < 3; ++j) std::cin >> draw(j);
            if (!std::cin) return 2;
            current_dm = servo.propose(phase, current_dm, wavelength,
                                       opd_per_dm_unit, dm_limit, draw);
            servo.update(current_dm, wavelength, opd_per_dm_unit);
            std::cout << current_dm.transpose() << '\n';
        }
        return 0;
    }
    if (mode != "controller") return 2;
    int history, future, count;
    std::cin >> history >> future >> count;
    if (history == 4 && future == 2) return run_controller<4, 2>(count);
    if (history == 30 && future == 3) return run_controller<30, 3>(count);
    if (history == 40 && future == 4) return run_controller<40, 4>(count);
    return 2;
}
