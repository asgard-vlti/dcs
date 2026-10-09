#include "../predictive_control.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using Clock = std::chrono::steady_clock;
    using heimdallr_ddspc::Modes;
    using heimdallr_ddspc::PredictiveControl;
    using heimdallr_ddspc::Telescopes;

    constexpr int warmup = 1200;
    constexpr int samples = 120000;
    constexpr double wavelength = 2.1;
    PredictiveControl<40, 4> controller;
    std::array<Telescopes, 2> lag{Telescopes::Zero(), Telescopes::Zero()};
    std::vector<double> microseconds;
    microseconds.reserve(samples);
    double checksum = 0.0;

    for (int i = 0; i < warmup + samples; ++i) {
        const double time = (i + 1) / 2000.0;
        Telescopes disturbance;
        for (int j = 0; j < 4; ++j) {
            disturbance(j) =
                (0.02 + 0.01 * j) * std::sin(2.0 * M_PI * 102.0 * time + j) +
                0.03 * std::sin(2.0 * M_PI * 24.0 * time + 0.5 * j);
        }
        const Telescopes residual = disturbance + lag[1];
        const Modes error = heimdallr_ddspc::to_modes(residual);
        const Modes draw = Modes::Zero();

        const auto start = Clock::now();
        controller.advance_regularization(i);
        const Modes proposed = controller.propose(error, draw);
        const Telescopes dm = heimdallr_ddspc::dm_command(proposed, wavelength)
                                  .cwiseMax(-0.4)
                                  .cwiseMin(0.4);
        const Modes applied =
            heimdallr_ddspc::applied_command_waves(dm, wavelength);
        controller.update(applied);
        const auto stop = Clock::now();

        if (!proposed.allFinite() || !dm.allFinite()) {
            std::cerr << "Non-finite controller output at frame " << i << '\n';
            return 2;
        }
        lag[1] = lag[0];
        lag[0] = dm * (6.0 / wavelength);
        checksum += proposed.squaredNorm();
        if (i >= warmup) {
            microseconds.push_back(
                std::chrono::duration<double, std::micro>(stop - start).count());
        }
    }

    std::sort(microseconds.begin(), microseconds.end());
    std::cout << std::setprecision(9)
              << "{\"history\":40,\"future\":4,\"samples\":" << samples
              << ",\"median_us\":" << microseconds[samples / 2]
              << ",\"p999_us\":" << microseconds[119879]
              << ",\"max_us\":" << microseconds.back()
              << ",\"checksum\":" << checksum << "}\n";
    return 0;
}
