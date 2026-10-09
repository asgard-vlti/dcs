#pragma once

#include <Eigen/Dense>

#include <array>
#include <cstdint>
#include <memory>

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
    bool continue_learning = true;
};

struct FreezeStatus {
    bool pending = false;
    bool frozen = false;
    const char* reason = nullptr;
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

    const std::array<double, Features * Features>& factor() const;

    double initial_covariance() const;

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

template <int HistoryLength = 30, int FutureLength = 3>
class PredictiveControl {
   public:
    static_assert(HistoryLength >= 2 && FutureLength >= 2);
    static constexpr int HistorySamples = HistoryLength;
    static constexpr int FutureSamples = FutureLength;
    static constexpr int Features = (FutureLength - 1 + 2 * HistoryLength) * 3;
    static constexpr int Outputs = FutureLength * 3;
    static constexpr int ControlFeatures = (2 * HistoryLength - 1) * 3;
    static constexpr int TrainingDelay = HistoryLength + FutureLength;

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

    void update(const Modes& applied_command, bool learn = true);

    void reset_tracking(const Modes& initial_command);

    void advance_regularization(int valid_frame_index);

    int iterations() const;
    int rls_updates() const;
    double regularization() const;
    const Parameters& parameters() const;
    const Modes& command() const;
    const ControlMatrix& predictive() const;
    const CorrelationMatrix& inverse() const;
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
    int rls_updates_ = 0;
    ControlVector past_;
    FeatureVector feature_;
    OutputVector target_;
    ControlMatrix predictive_;
    CorrelationMatrix inverse_;
    CorrelationMatrix correlation_;
    CrossMatrix cross_;
    CrossMatrix control_;
    Eigen::JacobiSVD<CorrelationMatrix> svd_;
};

extern template class PredictiveControl<4, 2>;
extern template class PredictiveControl<30, 3>;
extern template class PredictiveControl<40, 4>;
extern template class PredictiveControl<60, 5>;

struct ModelSnapshot {
    using Controller = PredictiveControl<>;

    Parameters parameters;
    int iterations = 0;
    int exploration_frames = 0;
    int rls_updates = 0;
    bool frozen = false;
    const char* freeze_reason = nullptr;
    int freeze_frame = 0;
    std::int64_t freeze_time_ns = 0;
    double regularization = 0.0;
    double initial_covariance = 0.0;
    std::int64_t model_time_ns = 0;
    bool trained = false;
    const char* source = "untrained";
    std::array<double, Controller::Features * Controller::Features> factor{};
    QrdRls<Controller::Features, Controller::Outputs>::WeightMatrix weights;
    Controller::CorrelationMatrix inverse;
    Controller::ControlMatrix predictive;
};

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

    void freeze_manual();
    bool frozen() const;
    const char* freeze_reason() const;

    int exploration_frames() const;
    const PredictiveControl<>& controller() const;
    std::unique_ptr<ModelSnapshot> snapshot_for_off();

   private:
    void capture(ModelSnapshot& snapshot) const;

    PredictiveControl<> controller_;
    std::unique_ptr<ModelSnapshot> last_trained_;
    bool last_trained_valid_ = false;
    std::int64_t last_update_ns_ = 0;
    double common_mode_ = 0.0;
    int exploration_frames_ = 0;
    bool active_ = false;
    bool frozen_ = false;
    const char* freeze_reason_ = nullptr;
    int freeze_frame_ = 0;
    std::int64_t freeze_time_ns_ = 0;

    void freeze(const char* reason);
};

}  // namespace heimdallr_ddspc
