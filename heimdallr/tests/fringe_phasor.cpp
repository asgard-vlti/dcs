#include "../fringe_phasor.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using heimdallr_fringe::FourBinKernel;
using heimdallr_fringe::pi;
using Complex = std::complex<double>;
using Pixel = std::array<double, 2>;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<double> constant_window(int size) {
    return std::vector<double>(size * size, 1.0);
}

std::vector<double> super_gaussian_window(int size) {
    std::vector<double> window(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const double radius =
                (std::pow(y - size / 2, 2) + std::pow(x - size / 2, 2)) /
                std::pow(size / 2, 2);
            window[y * size + x] = std::exp(-radius * radius);
        }
    }
    return window;
}

Complex direct_fft(const std::vector<double>& image, int size, int kx, int ky) {
    Complex value = 0.0;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const int shifted_x = (x + size / 2) % size;
            const int shifted_y = (y + size / 2) % size;
            const double angle = -2.0 * pi *
                (kx * shifted_x + ky * shifted_y) / size;
            value += image[y * size + x] * std::polar(1.0, angle);
        }
    }
    return value;
}

std::vector<Pixel> fft_for_kernel(
    const std::vector<double>& image, int size, const FourBinKernel& kernel) {
    std::vector<Pixel> ft(size * (size / 2 + 1));
    for (const auto& tap : kernel.taps) {
        const int kx = tap.index % (size / 2 + 1);
        const int ky = tap.index / (size / 2 + 1);
        const Complex value = direct_fft(image, size, kx, ky);
        ft[tap.index] = {value.real(), value.imag()};
    }
    return ft;
}

std::vector<double> fringe_image(
    int size, double fx, double fy, double phase, double shift_x = 0.0,
    bool with_envelope = false) {
    std::vector<double> image(size * size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const double xx = x - size / 2;
            const double yy = y - size / 2;
            const double fringe = 2.0 * std::cos(
                2.0 * pi * (fx * xx + fy * yy) / size + phase);
            const double envelope = !with_envelope ? 1.0 :
                std::exp(-((xx - shift_x) * (xx - shift_x) + yy * yy) /
                         (2.0 * 12.0 * 12.0));
            image[y * size + x] = fringe * envelope;
        }
    }
    return image;
}

double phase_error(Complex actual, double expected) {
    return std::arg(actual * std::polar(1.0, -expected));
}

void test_phase_and_amplitude() {
    constexpr int size = 64;
    const auto window = constant_window(size);
    Complex correlation[3][3];
    heimdallr_fringe::window_noise_correlations(window.data(), size, correlation);

    for (const auto& peak : {std::array<double, 2>{7.0, 11.0}, {7.35, 11.4}}) {
        const auto kernel = heimdallr_fringe::make_four_bin_kernel(
            size, peak[0], peak[1], correlation);
        if (peak[0] == 7.0) {
            check(std::abs(kernel.amplitude_response - 1.0) < 1e-12 &&
                  std::abs(kernel.noise_gain - 1.0) < 1e-12,
                  "integer peak did not preserve amplitude and noise");
        } else {
            double weight_square_sum = 0.0;
            for (const auto& tap : kernel.taps) {
                weight_square_sum += tap.weight * tap.weight;
            }
            check(std::abs(kernel.noise_gain - weight_square_sum /
                  (kernel.amplitude_response * kernel.amplitude_response)) < 1e-12,
                  "rectangular-window noise gain disagrees with independent bins");
        }
        const auto image = fringe_image(size, peak[0], peak[1], 0.7);
        const auto ft = fft_for_kernel(image, size, kernel);
        const Complex phasor = kernel.sample(ft.data(), 1.0);
        check(std::abs(phase_error(phasor, 0.7)) < 0.02,
              "fractional peak changed the piston phase");
        check(std::abs(std::abs(phasor) / (size * size) - 1.0) < 0.025,
              "sinc correction did not recover fringe amplitude");
        const Complex reversed = kernel.sample(ft.data(), -1.0);
        check(std::abs(reversed - std::conj(phasor)) < 1e-9,
              "baseline sign did not conjugate the phasor");
    }
}

void test_fft_boundaries() {
    constexpr int size = 32;
    const auto window = constant_window(size);
    Complex correlation[3][3];
    heimdallr_fringe::window_noise_correlations(window.data(), size, correlation);
    const auto image = fringe_image(size, 5.25, 8.3, 0.4);

    for (const auto& peak : {std::array<double, 2>{0.0, 0.0},
                             {0.2, 31.8}, {15.8, 8.3},
                             {16.0, 31.0}, {31.7, 31.6},
                             {-0.3, -0.4}}) {
        const auto kernel = heimdallr_fringe::make_four_bin_kernel(
            size, peak[0], peak[1], correlation);
        const auto ft = fft_for_kernel(image, size, kernel);
        const Complex sampled = kernel.sample(ft.data(), 1.0);
        Complex expected = 0.0;
        for (const auto& tap : kernel.taps) {
            expected += tap.weight * direct_fft(image, size,
                tap.x % size, tap.y % size);
        }
        expected /= kernel.amplitude_response;
        check(std::abs(sampled - expected) < 1e-7,
              "real FFT boundary mapping changed the complex value");
    }
}

void test_tip_tilt_coupling() {
    constexpr int size = 64;
    constexpr double fx = 7.35;
    constexpr double fy = 11.4;
    const auto window = constant_window(size);
    Complex correlation[3][3];
    heimdallr_fringe::window_noise_correlations(window.data(), size, correlation);
    const auto kernel = heimdallr_fringe::make_four_bin_kernel(
        size, fx, fy, correlation);
    double old_drift = 0.0;
    double new_drift = 0.0;
    for (double shift : {-4.0, -2.0, 2.0, 4.0}) {
        const auto image = fringe_image(size, fx, fy, 0.6, shift, true);
        const auto ft = fft_for_kernel(image, size, kernel);
        const Complex sampled = kernel.sample(ft.data(), 1.0);
        const Complex rounded = direct_fft(image, size, 7, 11);
        new_drift = std::max(new_drift, std::abs(phase_error(sampled, 0.6)));
        old_drift = std::max(old_drift, std::abs(phase_error(rounded, 0.6)));
    }
    check(new_drift < 0.5 * old_drift,
          "four-bin extraction did not reduce tip/tilt phase coupling");
}

void test_noise_gain() {
    constexpr int size = 32;
    const auto window = super_gaussian_window(size);
    Complex correlation[3][3];
    heimdallr_fringe::window_noise_correlations(window.data(), size, correlation);
    const auto kernel = heimdallr_fringe::make_four_bin_kernel(
        size, 7.4, 11.35, correlation);
    std::array<std::vector<Complex>, 4> factors;
    for (int tap = 0; tap < 4; ++tap) {
        factors[tap].reserve(size * size);
        const int kx = kernel.taps[tap].index % (size / 2 + 1);
        const int ky = kernel.taps[tap].index / (size / 2 + 1);
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const double angle = -2.0 * pi *
                    (kx * ((x + size / 2) % size) +
                     ky * ((y + size / 2) % size)) / size;
                factors[tap].push_back(window[y * size + x] *
                                       std::polar(1.0, angle));
            }
        }
    }

    std::mt19937 generator(12345);
    std::normal_distribution<double> gaussian(0.0, 1.0);
    double raw_power = 0.0;
    double sampled_power = 0.0;
    for (int frame = 0; frame < 1600; ++frame) {
        std::array<Complex, 4> bins{};
        for (int pixel = 0; pixel < size * size; ++pixel) {
            const double noise = gaussian(generator);
            for (int tap = 0; tap < 4; ++tap) {
                bins[tap] += noise * factors[tap][pixel];
            }
        }
        std::vector<Pixel> ft(size * (size / 2 + 1));
        for (int tap = 0; tap < 4; ++tap) {
            ft[kernel.taps[tap].index] = {bins[tap].real(), bins[tap].imag()};
        }
        raw_power += std::norm(bins[0]);
        sampled_power += std::norm(kernel.sample(ft.data(), 1.0));
    }
    check(std::abs(sampled_power / raw_power - kernel.noise_gain) < 0.1,
          "window covariance does not predict four-bin noise power");
}

void test_power_history() {
    heimdallr_fringe::PowerHistory<2, 4> history;
    check(history.v2(0) == 0.0, "empty history has nonzero V squared");
    for (int i = 1; i <= 4; ++i) {
        history.begin_frame(100.0, 2.0);
        history.record(0, i * 10.0 - 2.0);
        history.record(1, -2.0);
        history.end_frame();
    }
    check(history.count() == 4, "history did not fill");
    check(std::abs(history.v2(0) - 16.0 * 92.0 / 400.0) < 1e-12,
          "V squared did not average corrected phasor power");
    check(std::abs(history.noise_bias() - 2.0) < 1e-12,
          "noise bias average changed");
    const double before_skip = history.v2(0);
    check(history.v2(0) == before_skip, "skipped frame changed V squared");
    history.begin_frame(100.0, 2.0);
    history.record(0, 48.0);
    history.record(1, -2.0);
    history.end_frame();
    check(std::abs(history.v2(0) - 16.0 * 132.0 / 400.0) < 1e-12,
          "history rollover kept the oldest frame");

    heimdallr_fringe::PowerHistory<1, 4> dark_history;
    dark_history.begin_frame(0.0, 2.0);
    dark_history.record(0, 10.0);
    dark_history.end_frame();
    check(dark_history.v2(0) == 0.0,
          "zero DC denominator produced an invalid V squared");
}

int main() {
    try {
        test_phase_and_amplitude();
        test_fft_boundaries();
        test_tip_tilt_coupling();
        test_noise_gain();
        test_power_history();
        std::cout << "fringe phasor tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
