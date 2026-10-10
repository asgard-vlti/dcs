#include "../predictive_control.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

template <int HistoryLength, int FutureLength>
int benchmark(int samples) {
    using Clock = std::chrono::steady_clock;
    using heimdallr_ddspc::Modes;
    using heimdallr_ddspc::PredictiveControl;
    using heimdallr_ddspc::Telescopes;

    constexpr int warmup = 1200;
    constexpr double wavelength = 2.1;
    PredictiveControl<HistoryLength, FutureLength> controller;
    std::array<Telescopes, 2> lag{Telescopes::Zero(), Telescopes::Zero()};
    std::vector<double> control_us;
    std::vector<double> update_us;
    std::vector<double> total_us;
    control_us.reserve(samples);
    update_us.reserve(samples);
    total_us.reserve(samples);
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
        const auto control_done = Clock::now();
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
            control_us.push_back(std::chrono::duration<double, std::micro>(
                                     control_done - start)
                                     .count());
            update_us.push_back(std::chrono::duration<double, std::micro>(
                                    stop - control_done)
                                    .count());
            total_us.push_back(
                std::chrono::duration<double, std::micro>(stop - start).count());
        }
    }

    std::sort(control_us.begin(), control_us.end());
    std::sort(update_us.begin(), update_us.end());
    std::sort(total_us.begin(), total_us.end());
    std::cout << std::setprecision(9)
              << "{\"history\":" << decltype(controller)::HistorySamples
              << ",\"future\":" << decltype(controller)::FutureSamples
              << ",\"samples\":" << samples
              << ",\"median_us\":" << total_us[samples / 2]
              << ",\"p999_us\":" << total_us[(samples * 999 - 1) / 1000]
              << ",\"max_us\":" << total_us.back()
              << ",\"control_median_us\":" << control_us[samples / 2]
              << ",\"control_p999_us\":" << control_us[(samples * 999 - 1) / 1000]
              << ",\"control_max_us\":" << control_us.back()
              << ",\"update_median_us\":" << update_us[samples / 2]
              << ",\"update_p999_us\":" << update_us[(samples * 999 - 1) / 1000]
              << ",\"update_max_us\":" << update_us.back()
              << ",\"checksum\":" << checksum << "}\n";
    return 0;
}

int main(int argc, char** argv) {
    int history = 30;
    int future = 3;
    int samples = 120000;
    if (argc != 1 && argc != 3 && argc != 4) {
        std::cerr << "Usage: predictive_benchmark [history future [samples]]\n";
        return 2;
    }
    try {
        if (argc >= 3) {
            history = std::stoi(argv[1]);
            future = std::stoi(argv[2]);
        }
        if (argc == 4) samples = std::stoi(argv[3]);
    } catch (const std::exception&) {
        std::cerr << "Invalid benchmark argument\n";
        return 2;
    }
    if (samples <= 0) return 2;
    if (history == 20 && future == 2) return benchmark<20, 2>(samples);
    if (history == 30 && future == 3) return benchmark<30, 3>(samples);
    if (history == 40 && future == 3) return benchmark<40, 3>(samples);
    if (history == 50 && future == 3) return benchmark<50, 3>(samples);
    std::cerr << "Unsupported history preset\n";
    return 2;
}
