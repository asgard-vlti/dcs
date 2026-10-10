#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>

namespace heimdallr_fringe {

constexpr double pi = 3.14159265358979323846;

inline double sinc(double x) {
    return x == 0.0 ? 1.0 : std::sin(pi * x) / (pi * x);
}

inline double interpolation_response(double fraction) {
    return (1.0 - fraction) * sinc(fraction) +
           fraction * sinc(1.0 - fraction);
}

// Correlation of neighboring FFT bins for spatially white noise after the image window.
inline void window_noise_correlations(
    const double* window, int size, std::complex<double> (&correlation)[3][3]) {
    double window_power = 0.0;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            window_power += window[y * size + x] * window[y * size + x];
        }
    }
    if (!(window_power > 0.0)) {
        throw std::invalid_argument("FFT window has no power");
    }

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            std::complex<double> covariance = 0.0;
            for (int y = 0; y < size; ++y) {
                const int shifted_y = (y + size / 2) % size;
                for (int x = 0; x < size; ++x) {
                    const int shifted_x = (x + size / 2) % size;
                    const double weight = window[y * size + x] * window[y * size + x];
                    const double angle = -2.0 * pi *
                        (dx * shifted_x + dy * shifted_y) / size;
                    covariance += weight * std::polar(1.0, angle);
                }
            }
            correlation[dy + 1][dx + 1] = covariance / window_power;
        }
    }
}

struct FourBinKernel {
    struct Tap {
        std::size_t index;
        double weight;
        bool conjugate;
        int x;
        int y;
    };

    std::array<Tap, 4> taps{};
    double amplitude_response = 1.0;
    double noise_gain = 1.0;

    template <typename Bin>
    std::complex<double> sample(const Bin* ft, double sign) const {
        std::complex<double> weighted = 0.0;
        for (const Tap& tap : taps) {
            std::complex<double> value(ft[tap.index][0], ft[tap.index][1]);
            if (tap.conjugate) value = std::conj(value);
            weighted += tap.weight * value;
        }
        weighted /= amplitude_response;
        return {weighted.real(), weighted.imag() * sign};
    }
};

inline FourBinKernel make_four_bin_kernel(
    int size, double peak_x, double peak_y,
    const std::complex<double> (&correlation)[3][3]) {
    if (size < 2 || size % 2 != 0 ||
        !std::isfinite(peak_x) || !std::isfinite(peak_y)) {
        throw std::invalid_argument("Invalid Fourier peak geometry");
    }

    double x = std::fmod(peak_x, size);
    double y = std::fmod(peak_y, size);
    if (x < 0.0) x += size;
    if (y < 0.0) y += size;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const double dx = x - x0;
    const double dy = y - y0;
    FourBinKernel kernel;
    kernel.amplitude_response =
        interpolation_response(dx) * interpolation_response(dy);

    for (int row = 0; row < 2; ++row) {
        for (int col = 0; col < 2; ++col) {
            const int tap_x = x0 + col;
            const int tap_y = y0 + row;
            int stored_x = tap_x % size;
            int stored_y = tap_y % size;
            const bool conjugate = stored_x > size / 2;
            if (conjugate) {
                stored_x = size - stored_x;
                stored_y = (size - stored_y) % size;
            }
            kernel.taps[row * 2 + col] = {
                static_cast<std::size_t>(stored_y * (size / 2 + 1) + stored_x),
                (col == 0 ? 1.0 - dx : dx) *
                    (row == 0 ? 1.0 - dy : dy),
                conjugate, tap_x, tap_y};
        }
    }

    double weighted_noise = 0.0;
    for (const auto& left : kernel.taps) {
        for (const auto& right : kernel.taps) {
            const int delta_x = left.x - right.x;
            const int delta_y = left.y - right.y;
            weighted_noise += left.weight * right.weight *
                std::real(correlation[delta_y + 1][delta_x + 1]);
        }
    }
    kernel.noise_gain = weighted_noise /
        (kernel.amplitude_response * kernel.amplitude_response);
    if (!(kernel.noise_gain > 0.0) || !std::isfinite(kernel.noise_gain)) {
        throw std::invalid_argument("Invalid four-bin noise gain");
    }
    return kernel;
}

template <std::size_t Baselines, std::size_t Frames>
class PowerHistory {
public:
    void begin_frame(double dc_power, double noise_bias) {
        dc_sum_ += dc_power - dc_history_[next_];
        bias_sum_ += noise_bias - bias_history_[next_];
        dc_history_[next_] = dc_power;
        bias_history_[next_] = noise_bias;
    }

    void record(std::size_t baseline, double corrected_signal_power) {
        signal_sums_[baseline] +=
            corrected_signal_power - signal_history_[baseline][next_];
        signal_history_[baseline][next_] = corrected_signal_power;
    }

    void end_frame() {
        next_ = (next_ + 1) % Frames;
        if (count_ < Frames) ++count_;
    }

    double v2(std::size_t baseline) const {
        return dc_sum_ > 0.0 && std::isfinite(dc_sum_)
            ? 16.0 * signal_sums_[baseline] / dc_sum_ : 0.0;
    }

    double signal_power(std::size_t baseline) const {
        return count_ > 0 ? signal_sums_[baseline] / count_ : 0.0;
    }

    double noise_bias() const {
        return count_ > 0 ? bias_sum_ / count_ : 0.0;
    }

    std::size_t count() const { return count_; }

private:
    std::array<std::array<double, Frames>, Baselines> signal_history_{};
    std::array<double, Baselines> signal_sums_{};
    std::array<double, Frames> dc_history_{};
    std::array<double, Frames> bias_history_{};
    double dc_sum_ = 0.0;
    double bias_sum_ = 0.0;
    std::size_t next_ = 0;
    std::size_t count_ = 0;
};

}  // namespace heimdallr_fringe
