#include "../predictive_control.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using heimdallr_ddspc::Modes;
using heimdallr_ddspc::Telescopes;

struct Result {
    double residual_rms;
    double maximum_residual;
};

Result simulate(const std::vector<Telescopes>& disturbance, bool predictive) {
    constexpr double wavelength = 2.1;
    std::array<Telescopes, 2> lag{Telescopes::Zero(), Telescopes::Zero()};
    Telescopes current_dm = Telescopes::Zero();
    Modes integral_command = Modes::Zero();
    heimdallr_ddspc::DdspcServo servo;
    servo.enter();
    std::mt19937_64 generator(42);
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    double squared_residual = 0.0;
    double maximum_residual = 0.0;

    for (std::size_t i = 0; i < disturbance.size(); ++i) {
        Telescopes residual = disturbance[i] + lag[1];
        residual.array() -= residual.mean();
        if (i >= 2000) squared_residual += residual.squaredNorm();
        maximum_residual = std::max(maximum_residual,
                                    residual.cwiseAbs().maxCoeff());

        if (predictive) {
            Modes draw = Modes::Zero();
            if (i < 500) {
                for (int j = 0; j < 3; ++j) {
                    draw(j) = standard_normal(generator);
                }
            }
            current_dm = servo.propose(-residual, current_dm, wavelength, 6.0,
                                       0.4, draw);
            servo.update(current_dm, wavelength, 6.0);
        } else {
            integral_command -= 0.8 * heimdallr_ddspc::to_modes(residual);
            current_dm = heimdallr_ddspc::dm_command(integral_command,
                                                      wavelength)
                             .cwiseMax(-0.4)
                             .cwiseMin(0.4);
        }
        if (!current_dm.allFinite()) {
            std::cerr << "Non-finite command at frame " << i << '\n';
            return {NAN, NAN};
        }
        lag[1] = lag[0];
        lag[0] = current_dm * (6.0 / wavelength);
    }
    const double observations = 4.0 * (disturbance.size() - 2000);
    return {std::sqrt(squared_residual / observations), maximum_residual};
}

int main() {
    constexpr int samples = 3000;
    constexpr double dt = 1.0 / 2000.0;
    const std::array<double, 4> amplitude{0.07, 0.02, 0.07, 0.07};
    std::mt19937_64 generator(20260930);
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    std::array<double, 4> correlated{};
    const double decay = std::exp(-dt / 0.1);
    const double innovation_scale = 0.01 * std::sqrt(1.0 - decay * decay);
    std::vector<Telescopes> disturbance;
    disturbance.reserve(samples);
    double open_squared = 0.0;

    for (int i = 0; i < samples; ++i) {
        const double time = (i + 1) * dt;
        Telescopes sample;
        for (int j = 0; j < 4; ++j) {
            correlated[j] = decay * correlated[j] +
                            innovation_scale * standard_normal(generator);
            sample(j) = amplitude[j] *
                            std::sin(2.0 * M_PI * 102.0 * time + j) +
                        0.03 * std::sin(2.0 * M_PI * 24.0 * time + j + 0.5) +
                        correlated[j];
        }
        if (i >= 2000) {
            const Telescopes projected =
                sample.array() - sample.mean();
            open_squared += projected.squaredNorm();
        }
        disturbance.push_back(sample);
    }

    const double open_rms = std::sqrt(open_squared / (4.0 * (samples - 2000)));
    const Result predictive = simulate(disturbance, true);
    const Result integrator = simulate(disturbance, false);
    if (!std::isfinite(predictive.residual_rms)) return 2;
    std::cout << std::setprecision(8)
              << "{\"rate_hz\":2000,\"samples\":" << samples
              << ",\"open_rms\":" << open_rms
              << ",\"predictive_rms\":" << predictive.residual_rms
              << ",\"integrator_rms\":" << integrator.residual_rms
              << ",\"predictive_max_abs\":" << predictive.maximum_residual
              << ",\"integrator_max_abs\":" << integrator.maximum_residual
              << "}\n";
    return 0;
}
