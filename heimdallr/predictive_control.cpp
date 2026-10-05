#include "predictive_control.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace heimdallr_ddspc {

void validate(const Parameters& params) {
    if (!std::isfinite(params.reg_cutoff) || params.reg_cutoff <= 0.0) {
        throw std::invalid_argument("ddspc.reg_cutoff must be finite and positive");
    }
    if (!std::isfinite(params.reg_start) ||
        params.reg_start < params.reg_cutoff) {
        throw std::invalid_argument("ddspc.reg_start must be finite and at least reg_cutoff");
    }
    if (!std::isfinite(params.reg_divisor) || params.reg_divisor <= 1.0) {
        throw std::invalid_argument("ddspc.reg_divisor must be finite and greater than 1");
    }
    if (params.reg_interval <= 0) {
        throw std::invalid_argument("ddspc.reg_interval must be positive");
    }
    if (params.n_exploration < 0) {
        throw std::invalid_argument("ddspc.n_exploration must be nonnegative");
    }
    if (!std::isfinite(params.exploration_sigma) ||
        params.exploration_sigma < 0.0) {
        throw std::invalid_argument("ddspc.exploration_sigma must be finite and nonnegative");
    }
    if (!std::isfinite(params.gamma) || params.gamma <= 0.0 ||
        params.gamma > 1.0) {
        throw std::invalid_argument("ddspc.gamma must be finite and in (0, 1]");
    }
}

Modes to_modes(const Telescopes& telescopes) {
    return Modes(telescopes(0) - telescopes(2),
                 telescopes(1) - telescopes(2),
                 telescopes(3) - telescopes(2));
}

Telescopes to_telescopes(const Modes& modes) {
    const double mean = (modes(0) + modes(1) + modes(2)) / 4.0;
    return Telescopes(modes(0) - mean, modes(1) - mean, -mean,
                      modes(2) - mean);
}

Modes phase_error_modes(const Telescopes& phase_delay_waves) {
    return to_modes(-phase_delay_waves);
}

Telescopes dm_command(const Modes& command_waves, double wavelength,
                      double opd_per_dm_unit) {
    return to_telescopes(command_waves) * (wavelength / opd_per_dm_unit);
}

Modes applied_command_waves(const Telescopes& dm, double wavelength,
                            double opd_per_dm_unit) {
    return to_modes(dm * (opd_per_dm_unit / wavelength));
}

template <int Length>
void History<Length>::reset() {
    for (auto& item : data_) item.setZero();
    head_ = 0;
}

template <int Length>
void History<Length>::add(const Modes& item) {
    data_[head_] = item;
    head_ = (head_ + 1) % Length;
}

template <int Length>
auto History<Length>::get(int index) const -> const Modes& {
    int position = (head_ + index - 1) % Length;
    if (position < 0) position += Length;
    return data_[position];
}

template <int Features, int Outputs>
QrdRls<Features, Outputs>::QrdRls(double initial_covariance,
                                  double forgetting_factor)
    : initial_covariance_(initial_covariance),
      forgetting_factor_(forgetting_factor) {
    if (!std::isfinite(initial_covariance) ||
        !std::isfinite(forgetting_factor) || initial_covariance <= 0.0 ||
        forgetting_factor <= 0.0 || forgetting_factor > 1.0) {
        throw std::invalid_argument("Invalid QRD RLS initialization");
    }
    reset();
}

template <int Features, int Outputs>
void QrdRls<Features, Outputs>::reset() {
    r_.fill(0.0);
    for (int i = 0; i < Features; ++i) {
        r_[i * Features + i] = std::sqrt(initial_covariance_);
    }
    weights_.setZero();
}

template <int Features, int Outputs>
void QrdRls<Features, Outputs>::set_forgetting_factor(double value) {
    if (!std::isfinite(value) || value <= 0.0 || value > 1.0) {
        throw std::invalid_argument("Invalid QRD RLS forgetting factor");
    }
    forgetting_factor_ = value;
}

template <int Features, int Outputs>
void QrdRls<Features, Outputs>::update(const FeatureVector& feature,
                                       const OutputVector& target) {
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

template <int Features, int Outputs>
auto QrdRls<Features, Outputs>::weights() const -> const WeightMatrix& { return weights_; }

template <int Features, int Outputs>
auto QrdRls<Features, Outputs>::factor() const
    -> const std::array<double, Features * Features>& {
    return r_;
}

template <int Features, int Outputs>
double QrdRls<Features, Outputs>::initial_covariance() const {
    return initial_covariance_;
}

template <int Features, int Outputs>
double QrdRls<Features, Outputs>::gram(int i, int j) const {
    double value = 0.0;
    for (int k = 0; k < Features; ++k) {
        value += r_[k * Features + i] * r_[k * Features + j];
    }
    return value;
}

template <int HistoryLength, int FutureLength>
PredictiveControl<HistoryLength, FutureLength>::PredictiveControl(
    const Parameters& params)
    : rls_(1e-6, params.gamma), params_(params) {
    validate(params_);
    reset();
}

template <int HistoryLength, int FutureLength>
void PredictiveControl<HistoryLength, FutureLength>::configure(
    const Parameters& params) {
    validate(params);
    params_ = params;
    rls_.set_forgetting_factor(params.gamma);
    reset();
}

template <int HistoryLength, int FutureLength>
void PredictiveControl<HistoryLength, FutureLength>::reset(
    const Modes& initial_command) {
    rls_.reset();
    errors_.reset();
    commands_.reset();
    predictive_.setZero();
    inverse_.setZero();
    command_ = initial_command;
    previous_command_ = initial_command;
    regularization_ = params_.reg_start;
    iterations_ = 0;
}

template <int HistoryLength, int FutureLength>
Modes PredictiveControl<HistoryLength, FutureLength>::propose(
    const Modes& error, const Modes& normal_draw, bool exploration_enabled) {
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
    if (exploration_enabled && iterations_ < params_.n_exploration) {
        delta += (normal_draw * params_.exploration_sigma)
                     .cwiseMax(-0.4).cwiseMin(0.4);
    }
    delta = delta.cwiseMax(-0.5).cwiseMin(0.5);
    previous_command_ = command_;
    command_ += delta;
    return command_;
}

template <int HistoryLength, int FutureLength>
void PredictiveControl<HistoryLength, FutureLength>::update(
    const Modes& applied_command) {
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
    inverse_.setZero();
    const auto& singular = svd_.singularValues();
    const double threshold = singular(0) * 1e-15;
    for (int i = 0; i < Outputs; ++i) {
        if (singular(i) > threshold) {
            inverse_.noalias() +=
                (1.0 / singular(i)) * svd_.matrixV().col(i) *
                svd_.matrixU().col(i).transpose();
        }
    }
    control_.noalias() = -inverse_ * cross_;
    predictive_ = control_.template bottomRows<3>();
}

template <int HistoryLength, int FutureLength>
void PredictiveControl<HistoryLength, FutureLength>::advance_regularization(
    int valid_frame_index) {
    if (valid_frame_index % params_.reg_interval == 0 &&
        regularization_ > params_.reg_cutoff) {
        regularization_ = std::max(params_.reg_cutoff,
                                   regularization_ / params_.reg_divisor);
    }
}

template <int HistoryLength, int FutureLength>
int PredictiveControl<HistoryLength, FutureLength>::iterations() const {
    return iterations_;
}

template <int HistoryLength, int FutureLength>
double PredictiveControl<HistoryLength, FutureLength>::regularization() const {
    return regularization_;
}

template <int HistoryLength, int FutureLength>
auto PredictiveControl<HistoryLength, FutureLength>::parameters() const
    -> const Parameters& {
    return params_;
}

template <int HistoryLength, int FutureLength>
auto PredictiveControl<HistoryLength, FutureLength>::command() const
    -> const Modes& {
    return command_;
}

template <int HistoryLength, int FutureLength>
auto PredictiveControl<HistoryLength, FutureLength>::predictive() const
    -> const ControlMatrix& {
    return predictive_;
}

template <int HistoryLength, int FutureLength>
auto PredictiveControl<HistoryLength, FutureLength>::inverse() const
    -> const CorrelationMatrix& {
    return inverse_;
}

template <int HistoryLength, int FutureLength>
auto PredictiveControl<HistoryLength, FutureLength>::rls() const
    -> const QrdRls<Features, Outputs>& {
    return rls_;
}

void DdspcServo::enter(const Parameters& params) {
    controller_.configure(params);
    if (!last_trained_) last_trained_ = std::make_unique<ModelSnapshot>();
    last_trained_valid_ = false;
    last_update_ns_ = 0;
    active_ = false;
    exploration_frames_ = 0;
}

void DdspcServo::invalidate() {
    if (active_) {
        if (!last_trained_) last_trained_ = std::make_unique<ModelSnapshot>();
        if (controller_.iterations() > PredictiveControl<>::TrainingDelay) {
            capture(*last_trained_);
            last_trained_valid_ = true;
        }
        controller_.reset();
    }
    active_ = false;
}

void DdspcServo::capture(ModelSnapshot& snapshot) const {
    const auto& rls = controller_.rls();
    snapshot.parameters = controller_.parameters();
    snapshot.iterations = controller_.iterations();
    snapshot.exploration_frames = exploration_frames_;
    snapshot.regularization = controller_.regularization();
    snapshot.initial_covariance = rls.initial_covariance();
    snapshot.model_time_ns = last_update_ns_;
    snapshot.trained = snapshot.iterations > PredictiveControl<>::TrainingDelay;
    snapshot.factor = rls.factor();
    snapshot.weights = rls.weights();
    snapshot.inverse = controller_.inverse();
    snapshot.predictive = controller_.predictive();
}

std::unique_ptr<ModelSnapshot> DdspcServo::snapshot_for_off() {
    if (!last_trained_) last_trained_ = std::make_unique<ModelSnapshot>();
    if (active_ &&
        controller_.iterations() > PredictiveControl<>::TrainingDelay) {
        capture(*last_trained_);
        last_trained_->source = "active";
    } else if (last_trained_valid_) {
        last_trained_->source = "retained";
    } else {
        capture(*last_trained_);
        last_trained_->source = "untrained";
    }
    last_trained_valid_ = false;
    active_ = false;
    controller_.reset();
    return std::move(last_trained_);
}

Telescopes DdspcServo::propose(
    const Telescopes& phase_delay_waves, const Telescopes& current_dm,
    double wavelength, double opd_per_dm_unit, double dm_limit,
    const Modes& normal_draw) {
    if (!active_) {
        controller_.reset(
            applied_command_waves(current_dm, wavelength, opd_per_dm_unit));
        common_mode_ = current_dm.mean();
        last_update_ns_ = 0;
        active_ = true;
    }
    controller_.advance_regularization(controller_.iterations());
    const Modes command = controller_.propose(
        phase_error_modes(phase_delay_waves), normal_draw,
        exploration_frames_ < controller_.parameters().n_exploration);
    return (dm_command(command, wavelength, opd_per_dm_unit).array() +
            common_mode_)
        .matrix()
        .cwiseMax(-dm_limit)
        .cwiseMin(dm_limit);
}

void DdspcServo::update(const Telescopes& applied_dm, double wavelength,
                        double opd_per_dm_unit) {
    controller_.update(
        applied_command_waves(applied_dm, wavelength, opd_per_dm_unit));
    last_update_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    ++exploration_frames_;
}

int DdspcServo::exploration_frames() const { return exploration_frames_; }

auto DdspcServo::controller() const -> const PredictiveControl<>& { return controller_; }

template class QrdRls<1, 1>;
template class QrdRls<27, 6>;
template class QrdRls<249, 12>;
template class PredictiveControl<4, 2>;
template class PredictiveControl<40, 4>;

}  // namespace heimdallr_ddspc
