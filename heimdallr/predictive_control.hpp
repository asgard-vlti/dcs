#pragma once

#include <Eigen/Dense>

#include <array>

namespace heimdallr_ddspc {

using Modes = Eigen::Vector3d;
using Telescopes = Eigen::Vector4d;

struct Parameters {
    double reg_start = 1e5;
    double reg_cutoff = 0.2;
    double reg_divisor = 5.0;
    int reg_interval = 100;
    int n_exploration = 500;
    double exploration_sigma = 0.01;
    double gamma = 1.0;
};

void validate(const Parameters& params);

Modes to_modes(const Telescopes& telescopes);

Telescopes to_telescopes(const Modes& modes);

Modes phase_error_modes(const Telescopes& phase_delay_waves);

Telescopes dm_command(const Modes& command_waves, double wavelength,
                      double opd_per_dm_unit = 6.0);

Modes applied_command_waves(const Telescopes& dm, double wavelength,
                            double opd_per_dm_unit = 6.0);

template <int Length>
class History {
   public:
    void reset();

    void add(const Modes& item);

    const Modes& get(int index) const;

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

    explicit QrdRls(double initial_covariance, double forgetting_factor = 1.0);

    void reset();

    void set_forgetting_factor(double value);

    void update(const FeatureVector& feature, const OutputVector& target);

    const WeightMatrix& weights() const;

    double gram(int i, int j) const;

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

    explicit PredictiveControl(const Parameters& params = Parameters{});

    void configure(const Parameters& params);

    void reset(const Modes& initial_command = Modes::Zero());

    Modes propose(const Modes& error, const Modes& normal_draw,
                  bool exploration_enabled = true);

    void update(const Modes& applied_command);

    void advance_regularization(int valid_frame_index);

    int iterations() const;
    double regularization() const;
    const Parameters& parameters() const;
    const Modes& command() const;
    const ControlMatrix& predictive() const;
    const QrdRls<Features, Outputs>& rls() const;

   private:
    QrdRls<Features, Outputs> rls_;
    Parameters params_;
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

extern template class PredictiveControl<4, 2>;
extern template class PredictiveControl<40, 4>;

class DdspcServo {
   public:
    void enter(const Parameters& params = Parameters{});

    void invalidate();

    Telescopes propose(const Telescopes& phase_delay_waves,
                       const Telescopes& current_dm, double wavelength,
                       double opd_per_dm_unit, double dm_limit,
                       const Modes& normal_draw);

    void update(const Telescopes& applied_dm, double wavelength,
                double opd_per_dm_unit);

    int exploration_frames() const;
    const PredictiveControl<>& controller() const;

   private:
    PredictiveControl<> controller_;
    double common_mode_ = 0.0;
    int exploration_frames_ = 0;
    bool active_ = false;
};

}  // namespace heimdallr_ddspc
