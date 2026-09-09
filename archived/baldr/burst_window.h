#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>

// Used to hold a fixed burst window with idx=0 just after reset ad idx=-1 at the end of the burst
// we update samples iteratively and can use this window to esimate global slopes and apply rolling 
// regression or kalman estimates of instanteous values (i.e. intensity)
// Also (could) include methods to offset reset anomalies etc 


class BurstWindow {
public:
    int nbread = 0;                             // Number of samples in a burst
    double fps = 0.0;                           // Frame rate [Hz]
    double phase = 0.0;                         // Optional phase offset
    size_t nchannels = 140;                     // Number of channels (rows) - number of DM actuators
    std::vector<double> tgrid;                  // Time axis (fixed, size = nbread)
    std::vector<std::vector<double>> values;    // values[channel][time_index]
    size_t idx = 0;                             // Rolling time index
    std::string label = "";                     // Optional label

    BurstWindow() = default;

    // Configure the burst window
    void configure(int nbread_in, double fps_in, double phase_in = 0.0, size_t nchannels_in = 140) {
        if (nbread_in <= 0 || fps_in <= 0.0 || nchannels_in == 0)
            throw std::invalid_argument("nbread, fps, and nchannels must be positive");

        nbread = nbread_in;
        fps = fps_in;
        phase = phase_in;
        nchannels = nchannels_in;

        const double dt = 1.0 / fps;
        tgrid.resize(nbread);
        for (int i = 0; i < nbread; ++i)
            tgrid[i] = phase + i * dt;

        values.assign(nchannels, std::vector<double>(nbread, 0.0));
        idx = 0;
    }

    // Update one new sample (vector of N values at current tgrid[idx])
    void update_value(const std::vector<double>& sample) {
        if (nbread == 0)
            throw std::runtime_error("BurstWindow not configured");
        if (sample.size() != nchannels)
            throw std::invalid_argument("Sample size does not match number of channels");

        for (size_t ch = 0; ch < nchannels; ++ch)
            values[ch][idx] = sample[ch];

        idx = (idx + 1) % nbread;
    }

    void update_idx(size_t new_idx) {
        if (new_idx >= nbread)
            throw std::out_of_range("Index exceeds burst size");
        idx = new_idx;
    }

    void reset() {
        for (auto& row : values)
            std::fill(row.begin(), row.end(), 0.0);
        idx = 0;
    }

    // Get current time and sample vector across all channels at current idx
    std::pair<double, std::vector<double>> current_sample() const {
        std::vector<double> sample(nchannels);
        for (size_t ch = 0; ch < nchannels; ++ch)
            sample[ch] = values[ch][idx];
        return {tgrid[idx], sample};
    }

    // Return time-ordered values for a given channel (0-based)
    std::vector<double> get_ordered_values(size_t channel) const {
        if (channel >= nchannels)
            throw std::out_of_range("Invalid channel index");
        std::vector<double> ordered(nbread);
        for (int i = 0; i < nbread; ++i)
            ordered[i] = values[channel][(idx + i) % nbread];
        return ordered;
    }

    void print_summary() const {
        std::cout << "BurstWindow: nbread = " << nbread
                  << ", fps = " << fps
                  << ", nchannels = " << nchannels
                  << ", phase = " << phase
                  << ", tgrid[0] = " << (tgrid.empty() ? 0.0 : tgrid[0])
                  << ", tgrid[end] = " << (tgrid.empty() ? 0.0 : tgrid.back())
                  << ", idx = " << idx
                  << std::endl;
    }
};



enum class WindowMode {
    RAMP,   // full y = m t + b
    MEAN    // constant y = b only
};



struct IntensityEstimator {
    virtual void reset() = 0;
    virtual void process(double t, double y) = 0;
    virtual double predict(double t_eval) const = 0;
    virtual ~IntensityEstimator() = default;
};



struct KalmanFilterSlope : public IntensityEstimator {
    WindowMode mode = WindowMode::RAMP;
    Eigen::Vector2d x;
    Eigen::Matrix2d P;
    Eigen::Matrix2d Q;
    double R;

    KalmanFilterSlope(WindowMode mode_ = WindowMode::RAMP)
        : mode(mode_) {
        x.setZero(); P.setIdentity(); Q.setIdentity(); R = 1.0;
    }

    void reset() override {
        x.setZero(); P.setIdentity();
    }

    void process(double t, double y) override {
        Eigen::RowVector2d H = (mode == WindowMode::RAMP) ? Eigen::RowVector2d(t, 1.0) : Eigen::RowVector2d(0.0, 1.0);
        P += Q;
        double S = H * P * H.transpose() + R;
        Eigen::Vector2d K = P * H.transpose() / S;
        x += K * (y - H * x);
        P -= K * H * P;
    }

    double predict(double t_eval) const override {
        return (mode == WindowMode::RAMP) ? x[0] * t_eval + x[1] : x[1];
    }
};



struct BoxcarFilter : public IntensityEstimator {
    int N = 10;
    std::vector<double> buffer;
    int write_idx = 0;
    int count = 0;
    double sum = 0.0;

    BoxcarFilter(int window_size = 10) : N(window_size), buffer(window_size, 0.0) {}

    void reset() override {
        std::fill(buffer.begin(), buffer.end(), 0.0);
        write_idx = 0;
        sum = 0.0;
        count = 0;
    }

    void process(double /*t*/, double y) override {
        double old = buffer[write_idx];
        buffer[write_idx] = y;
        sum += y - old;
        write_idx = (write_idx + 1) % N;
        if (count < N) count++;
    }

    double predict(double /*t_eval*/) const override {
        return (count == 0) ? 0.0 : sum / count;
    }
};







// class BurstWindow {
// public:
//     int nbread = 0;             // Number of samples in a burst
//     double fps = 0.0;           // Frame rate [Hz]
//     double phase = 0.0;         // Optional phase offset
//     std::vector<double> tgrid;  // Time axis (fixed)
//     std::vector<double> values; // Rolling value buffer
//     size_t idx = 0;             // Rolling index
//     std::string label = "";     // Optional label

//     BurstWindow() = default;

//     void configure(int nbread_in, double fps_in, double phase_in = 0.0) {
//         if (nbread_in <= 0 || fps_in <= 0.0)
//             throw std::invalid_argument("nbread and fps must be positive");

//         nbread = nbread_in;
//         fps = fps_in;
//         phase = phase_in;

//         double dt = 1.0 / fps;
//         tgrid.resize(nbread);
//         values.assign(nbread, 0.0);  // Initialize all to 0
//         for (int i = 0; i < nbread; ++i)
//             tgrid[i] = phase + i * dt;

//         idx = 0;
//     }

//     void update_value(double val) {
//         if (nbread == 0)
//             throw std::runtime_error("BurstWindow not configured");

//         values[idx] = val;
//         idx = (idx + 1) % nbread;  // Wrap around
//     }

//     void update_idx(size_t new_idx) {
//         idx = new_idx;
//     }

//     void reset() {
//         std::fill(values.begin(), values.end(), 0.0);
//         idx = 0;
//     }

//     std::pair<double, double> current_sample() const {
//         return {tgrid[idx], values[idx]};
//     }

//     std::vector<double> get_ordered_values() const {
//         std::vector<double> ordered(nbread);
//         for (int i = 0; i < nbread; ++i)
//             ordered[i] = values[(idx + i) % nbread];
//         return ordered;
//     }

//     void print_summary() const {
//         std::cout << "BurstWindow: nbread = " << nbread
//                   << ", fps = " << fps
//                   << ", phase = " << phase
//                   << ", tgrid[0] = " << (tgrid.empty() ? 0.0 : tgrid[0])
//                   << ", tgrid[" << nbread - 1 << "] = " << (tgrid.empty() ? 0.0 : tgrid.back())
//                   << std::endl;
//     }
// };

// // class BurstWindow {
// // public:
// //     // Number of samples per burst (e.g., from "nbreadworeset")
// //     int nbread = 0;

// //     // Frame rate in Hz
// //     double fps = 0.0;

// //     // Optional phase offset (time shift)
// //     double phase = 0.0;

// //     // Time grid [s] relative to the burst start
// //     std::vector<double> tgrid;

// //     // Optional burst label or tag
// //     std::string label = "";

// //     // Constructor
// //     BurstWindow() = default;

// //     // Configures the burst window (replaces time grid)
// //     void configure(int nbread_in, double fps_in, double phase_in = 0.0) {
// //         if (nbread_in <= 0 || fps_in <= 0.0)
// //             throw std::invalid_argument("nbread and fps must be positive");

// //         nbread = nbread_in;
// //         fps = fps_in;
// //         phase = phase_in;

// //         const double dt = 1.0 / fps;
// //         tgrid.resize(nbread);
// //         for (int i = 0; i < nbread; ++i)
// //             tgrid[i] = phase + i * dt;
// //     }

// //     // Debug print
// //     void print_summary() const {
// //         std::cout << "BurstWindow: nbread = " << nbread
// //                   << ", fps = " << fps
// //                   << ", phase = " << phase
// //                   << ", tgrid[0] = " << (tgrid.empty() ? 0.0 : tgrid[0])
// //                   << ", tgrid[" << nbread - 1 << "] = " << (tgrid.empty() ? 0.0 : tgrid.back())
// //                   << std::endl;
// //     }
// // };