#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace heimdallr_ddspc {

using Modes = Eigen::Vector3d;
using Telescopes = Eigen::Vector4d;

inline Modes to_modes(const Telescopes& telescopes) {
    return Modes(telescopes(0) - telescopes(2),
                 telescopes(1) - telescopes(2),
                 telescopes(3) - telescopes(2));
}

inline Telescopes to_telescopes(const Modes& modes) {
    const double mean = (modes(0) + modes(1) + modes(2)) / 4.0;
    return Telescopes(modes(0) - mean, modes(1) - mean, -mean,
                      modes(2) - mean);
}

inline Modes phase_error_modes(const Telescopes& phase_delay_waves) {
    return to_modes(-phase_delay_waves);
}

inline Telescopes dm_command(const Modes& command_waves, double wavelength,
                             double opd_per_dm_unit = 6.0) {
    return to_telescopes(command_waves) * (wavelength / opd_per_dm_unit);
}

inline Modes applied_command_waves(const Telescopes& dm, double wavelength,
                                   double opd_per_dm_unit = 6.0) {
    return to_modes(dm * (opd_per_dm_unit / wavelength));
}

template <int Length>
class History {
   public:
    void reset() {
        for (auto& item : data_) item.setZero();
        head_ = 0;
    }

    void add(const Modes& item) {
        data_[head_] = item;
        head_ = (head_ + 1) % Length;
    }

    const Modes& get(int index) const {
        int position = (head_ + index - 1) % Length;
        if (position < 0) position += Length;
        return data_[position];
    }

   private:
    std::array<Modes, Length> data_{};
    int head_ = 0;
};

template <int Features, int Outputs>
class QrdRls {
   public:
    using FeatureVector = Eigen::Matrix<double, Features, 1>;
    using OutputVector = Eigen::Matrix<double, Outputs, 1>;
    using WeightMatrix = Eigen::Matrix<double, Features, Outputs>;

    explicit QrdRls(double initial_covariance, double forgetting_factor = 1.0)
        : initial_covariance_(initial_covariance),
          forgetting_factor_(forgetting_factor) {
        if (initial_covariance <= 0.0 || forgetting_factor <= 0.0 ||
            forgetting_factor > 1.0) {
            throw std::invalid_argument("Invalid QRD RLS initialization");
        }
        reset();
    }

    void reset() {
        r_.fill(0.0);
        for (int i = 0; i < Features; ++i) {
            r_[i * Features + i] = std::sqrt(initial_covariance_);
        }
        weights_.setZero();
    }

    void update(const FeatureVector& feature, const OutputVector& target) {
        const double root_forgetting = std::sqrt(forgetting_factor_);
        if (root_forgetting != 1.0) {
            for (double& value : r_) value *= root_forgetting;
        }
        for (int i = 0; i < Features; ++i) row_[i] = feature(i);

        // Triangularize [sqrt(lambda) R; x^T] using only its nonzero last row.
        for (int k = 0; k < Features; ++k) {
            const double a = r_[k * Features + k];
            const double b = row_[k];
            if (b == 0.0) continue;
            const double radius = std::hypot(a, b);
            const double cosine = a / radius;
            const double sine = -b / radius;
            for (int j = k; j < Features; ++j) {
                const double upper = r_[k * Features + j];
                const double lower = row_[j];
                r_[k * Features + j] = cosine * upper - sine * lower;
                row_[j] = sine * upper + cosine * lower;
            }
        }

        const OutputVector residual = target - weights_.transpose() * feature;
        for (int i = 0; i < Features; ++i) {
            double value = feature(i);
            for (int j = 0; j < i; ++j) {
                value -= r_[j * Features + i] * z_[j];
            }
            z_[i] = value / r_[i * Features + i];
        }
        for (int i = Features - 1; i >= 0; --i) {
            double value = z_[i];
            for (int j = i + 1; j < Features; ++j) {
                value -= r_[i * Features + j] * gain_[j];
            }
            gain_[i] = value / r_[i * Features + i];
        }
        for (int i = 0; i < Features; ++i) {
            for (int j = 0; j < Outputs; ++j) {
                weights_(i, j) += gain_[i] * residual(j);
            }
        }
    }

    const WeightMatrix& weights() const { return weights_; }

    double gram(int i, int j) const {
        double value = 0.0;
        for (int k = 0; k < Features; ++k) {
            value += r_[k * Features + i] * r_[k * Features + j];
        }
        return value;
    }

   private:
    double initial_covariance_;
    double forgetting_factor_;
    std::array<double, Features * Features> r_{};
    std::array<double, Features> row_{};
    std::array<double, Features> z_{};
    std::array<double, Features> gain_{};
    WeightMatrix weights_;
};

template <int HistoryLength = 40, int FutureLength = 4>
class PredictiveControl {
   public:
    static_assert(HistoryLength >= 2 && FutureLength >= 2);
    static constexpr int Features = (FutureLength - 1 + 2 * HistoryLength) * 3;
    static constexpr int Outputs = FutureLength * 3;
    static constexpr int ControlFeatures = (2 * HistoryLength - 1) * 3;

    using FeatureVector = Eigen::Matrix<double, Features, 1>;
    using OutputVector = Eigen::Matrix<double, Outputs, 1>;
    using ControlVector = Eigen::Matrix<double, ControlFeatures, 1>;
    using ControlMatrix = Eigen::Matrix<double, 3, ControlFeatures>;
    using CorrelationMatrix = Eigen::Matrix<double, Outputs, Outputs>;
    using CrossMatrix = Eigen::Matrix<double, Outputs, ControlFeatures>;

    PredictiveControl()
        : rls_(1e-6, 1.0) {
        reset();
    }

    void reset(const Modes& initial_command = Modes::Zero()) {
        rls_.reset();
        errors_.reset();
        commands_.reset();
        predictive_.setZero();
        command_ = initial_command;
        previous_command_ = initial_command;
        regularization_ = 1e5;
        iterations_ = 0;
    }

    Modes propose(const Modes& error, const Modes& normal_draw,
                  bool exploration_enabled = true) {
        errors_.add(error);
        int offset = 0;
        for (int i = 0; i < HistoryLength - 1; ++i) {
            past_.template segment<3>(offset) = commands_.get(-i);
            offset += 3;
        }
        for (int i = 0; i < HistoryLength; ++i) {
            past_.template segment<3>(offset) = errors_.get(-i);
            offset += 3;
        }
        Modes delta = predictive_ * past_ - 0.2 * error;
        if (exploration_enabled && iterations_ < 500) {
            delta += (normal_draw * 0.01).cwiseMax(-0.4).cwiseMin(0.4);
        }
        delta = delta.cwiseMax(-0.5).cwiseMin(0.5);
        previous_command_ = command_;
        command_ += delta;
        return command_;
    }

    void update(const Modes& applied_command) {
        ++iterations_;
        commands_.add(applied_command - previous_command_);
        if (iterations_ <= HistoryLength + FutureLength) return;

        int offset = 0;
        for (int i = 1; i < FutureLength; ++i) {
            feature_.template segment<3>(offset) = commands_.get(-i);
            offset += 3;
        }
        for (int i = 0; i < HistoryLength; ++i) {
            feature_.template segment<3>(offset) = commands_.get(-(i + FutureLength));
            offset += 3;
        }
        for (int i = 0; i < HistoryLength; ++i) {
            feature_.template segment<3>(offset) = errors_.get(-(i + FutureLength));
            offset += 3;
        }
        for (int i = 0; i < FutureLength; ++i) {
            target_.template segment<3>(3 * i) = errors_.get(-i);
        }
        rls_.update(feature_, target_);

        const auto& weight = rls_.weights();
        correlation_.noalias() =
            weight.template topRows<Outputs>() *
            weight.template topRows<Outputs>().transpose();
        cross_.noalias() =
            weight.template topRows<Outputs>() *
            weight.template bottomRows<ControlFeatures>().transpose();
        const double scale =
            std::max(correlation_.cwiseAbs().maxCoeff(),
                     cross_.cwiseAbs().maxCoeff());
        correlation_.diagonal().array() += scale * regularization_;
        svd_.compute(correlation_, Eigen::ComputeFullU | Eigen::ComputeFullV);
        CorrelationMatrix inverse = CorrelationMatrix::Zero();
        const auto& singular = svd_.singularValues();
        const double threshold = singular(0) * 1e-15;
        for (int i = 0; i < Outputs; ++i) {
            if (singular(i) > threshold) {
                inverse.noalias() +=
                    (1.0 / singular(i)) * svd_.matrixV().col(i) *
                    svd_.matrixU().col(i).transpose();
            }
        }
        control_.noalias() = -inverse * cross_;
        predictive_ = control_.template bottomRows<3>();
    }

    void advance_regularization(int valid_frame_index) {
        if (valid_frame_index % 100 == 0 && regularization_ > 0.2) {
            regularization_ /= 5.0;
        }
    }

    int iterations() const { return iterations_; }
    double regularization() const { return regularization_; }
    const Modes& command() const { return command_; }
    const ControlMatrix& predictive() const { return predictive_; }
    const QrdRls<Features, Outputs>& rls() const { return rls_; }

   private:
    QrdRls<Features, Outputs> rls_;
    History<HistoryLength + FutureLength> errors_;
    History<HistoryLength + FutureLength> commands_;
    Modes command_ = Modes::Zero();
    Modes previous_command_ = Modes::Zero();
    double regularization_ = 1e5;
    int iterations_ = 0;
    ControlVector past_;
    FeatureVector feature_;
    OutputVector target_;
    ControlMatrix predictive_;
    CorrelationMatrix correlation_;
    CrossMatrix cross_;
    CrossMatrix control_;
    Eigen::JacobiSVD<CorrelationMatrix> svd_;
};

extern template class PredictiveControl<40, 4>;

class DdspcServo {
   public:
    void enter() {
        controller_.reset();
        active_ = false;
        exploration_frames_ = 0;
    }

    void invalidate() {
        if (active_) controller_.reset();
        active_ = false;
    }

    Telescopes propose(const Telescopes& phase_delay_waves,
                       const Telescopes& current_dm, double wavelength,
                       double opd_per_dm_unit, double dm_limit,
                       const Modes& normal_draw) {
        if (!active_) {
            controller_.reset(
                applied_command_waves(current_dm, wavelength, opd_per_dm_unit));
            common_mode_ = current_dm.mean();
            active_ = true;
        }
        controller_.advance_regularization(controller_.iterations());
        const Modes command = controller_.propose(
            phase_error_modes(phase_delay_waves), normal_draw,
            exploration_frames_ < 500);
        return (dm_command(command, wavelength, opd_per_dm_unit).array() +
                common_mode_)
            .matrix()
            .cwiseMax(-dm_limit)
            .cwiseMin(dm_limit);
    }

    void update(const Telescopes& applied_dm, double wavelength,
                double opd_per_dm_unit) {
        controller_.update(
            applied_command_waves(applied_dm, wavelength, opd_per_dm_unit));
        ++exploration_frames_;
    }

    int exploration_frames() const { return exploration_frames_; }
    const PredictiveControl<>& controller() const { return controller_; }

   private:
    PredictiveControl<> controller_;
    double common_mode_ = 0.0;
    int exploration_frames_ = 0;
    bool active_ = false;
};

}  // namespace heimdallr_ddspc
