
// // #include "controller.h"
// // #include <iostream>
// // #include <algorithm>
// // #include <stdexcept>
// #include "controller.h"
// #include <algorithm>
// #include <stdexcept>

// // on my mac: clang++ -std=c++17 -Wall -I/opt/homebrew/opt/eigen/include/eigen3 -o controller_exec controller.cpp


// // PIDController Implementation
// // called it PIDController_1 due to conflicting name in baldr.cpp - this should eventually be overwritten by this abstract class version once stable!

// // If you use the PID ctor from a baldr config, include it here to avoid header cycles.
// // Comment this out if you don't need the bdr_controller constructor.
// // #include "baldr.h"

#include "controller.h"
#include <algorithm>
#include <stdexcept>

// If you want PIDController_1(bdr_controller) ctor, include baldr.h and enable the ctor below.
// #include "baldr.h"

// ------------------------------ Utilities -----------------------------------
namespace {

// Add this small normalizer at the top of the file (or inside an anonymous namespace).
static std::string leaky_norm_key(std::string name) {
    std::string low = name;
    for (auto &c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (low == "k" || low == "ki")            return "K";          // gain vector
    if (low == "alpha" || low == "rho" ||
        low == "leak" || low == "lambda")     return "alpha";      // leak factor
    if (low == "setpoint")                    return "set_point";
    if (low == "integral")                    return "integral";
    if (low == "output")                      return "output";
    if (low == "prev_error" || low == "preverror" || low == "prev")
                                              return "prev_error";
    // Keep exact name for canonical keys like "K"
    return name;
}


inline void ensure_size(Eigen::VectorXd& v, std::ptrdiff_t n, double fill = 0.0) {
    if (v.size() != n) v = Eigen::VectorXd::Constant(n, fill);
}
inline std::ptrdiff_t infer_size(std::initializer_list<std::reference_wrapper<const Eigen::VectorXd>> xs) {
    std::ptrdiff_t n = 0;
    for (const auto& r : xs) n = std::max<std::ptrdiff_t>(n, r.get().size());
    return n;
}
inline void clamp_to_limits(Eigen::VectorXd& v, const Eigen::VectorXd& lo, const Eigen::VectorXd& hi) {
    if (v.size() == lo.size() && v.size() == hi.size()) {
        for (Eigen::Index i = 0; i < v.size(); ++i) {
            if (v(i) < lo(i)) v(i) = lo(i);
            if (v(i) > hi(i)) v(i) = hi(i);
        }
    }
}
} // namespace



// =============================== PIDController_1 =============================

PIDController_1::PIDController_1() {
    const std::ptrdiff_t n = 140;
    kp           = Eigen::VectorXd::Ones(n);
    ki           = Eigen::VectorXd::Zero(n);
    kd           = Eigen::VectorXd::Zero(n);
    lower_limits = -Eigen::VectorXd::Ones(n);
    upper_limits =  Eigen::VectorXd::Ones(n);
    set_point    = Eigen::VectorXd::Zero(n);
    output       = Eigen::VectorXd::Zero(n);
    integrals    = Eigen::VectorXd::Zero(n);
    prev_errors  = Eigen::VectorXd::Zero(n);
    ctrl_type    = "PID";
}

PIDController_1::PIDController_1(const Eigen::VectorXd& Kp,
                                 const Eigen::VectorXd& Ki,
                                 const Eigen::VectorXd& Kd,
                                 const Eigen::VectorXd& lower,
                                 const Eigen::VectorXd& upper,
                                 const Eigen::VectorXd& setpoint) {
    const std::ptrdiff_t n = std::max({Kp.size(), Ki.size(), Kd.size(),
                                       lower.size(), upper.size(), setpoint.size()});
    ensure_size(kp, n, 1.0);     kp = Kp;
    ensure_size(ki, n, 0.0);     ki = Ki;
    ensure_size(kd, n, 0.0);     kd = Kd;
    ensure_size(lower_limits, n, -1.0); lower_limits = lower;
    ensure_size(upper_limits, n,  1.0); upper_limits = upper;
    ensure_size(set_point, n, 0.0);     set_point    = setpoint;

    output      = Eigen::VectorXd::Zero(n);
    integrals   = Eigen::VectorXd::Zero(n);
    prev_errors = Eigen::VectorXd::Zero(n);
    ctrl_type   = "PID";
}

// Uncomment if you included baldr.h above.
/*
PIDController_1::PIDController_1(const bdr_controller& cfg)
    : PIDController_1(cfg.kp, cfg.ki, cfg.kd, cfg.lower_limits, cfg.upper_limits, cfg.set_point) {}
*/

void PIDController_1::ensure_sizes(std::ptrdiff_t n) {
    ensure_size(kp, n, 1.0);
    ensure_size(ki, n, 0.0);
    ensure_size(kd, n, 0.0);
    ensure_size(lower_limits, n, -1.0);
    ensure_size(upper_limits, n,  1.0);
    ensure_size(set_point, n, 0.0);
    ensure_size(output, n, 0.0);
    ensure_size(integrals, n, 0.0);
    ensure_size(prev_errors, n, 0.0);
}
void PIDController_1::clamp_vector(Eigen::VectorXd& v) const {
    clamp_to_limits(v, lower_limits, upper_limits);
}

Eigen::VectorXd PIDController_1::process(const Eigen::VectorXd& measured) {
    const std::ptrdiff_t n = std::max(set_point.size(), measured.size());
    ensure_sizes(n);

    const Eigen::VectorXd error = measured - set_point; //set_point - measured;

    // I += Ki ∘ e (digital); anti-windup clamp
    integrals.array() += (ki.array() * error.array());
    clamp_vector(integrals);

    // D = Kd ∘ (e - e_prev)
    const Eigen::VectorXd deriv = kd.cwiseProduct(error - prev_errors);

    // P = Kp ∘ e
    const Eigen::VectorXd prop  = kp.cwiseProduct(error);

    // u = P + I + D ; soft clamp
    output = prop + integrals + deriv;
    clamp_vector(output);

    prev_errors = error;
    return output;
}

void PIDController_1::reset() {
    output.setZero();
    integrals.setZero();
    prev_errors.setZero();
}

void PIDController_1::set_all_gains_to_zero() {
    kp.setZero(); ki.setZero(); kd.setZero();
}

void PIDController_1::set_parameter(const std::string& name, const Param& value) {
    if (std::holds_alternative<Vec>(value)) {
        const auto& v = std::get<Vec>(value);
        if (name == "kp")              kp = v;
        else if (name == "ki")         ki = v;
        else if (name == "kd")         kd = v;
        else if (name == "lower_limits") lower_limits = v;
        else if (name == "upper_limits") upper_limits = v;
        else if (name == "set_point")    set_point    = v;
        else if (name == "output")       output       = v;
        else if (name == "integrals")    integrals    = v;
        else if (name == "prev_errors")  prev_errors  = v;
        else throw std::invalid_argument("PID::set_parameter unknown vector: " + name);

        const std::ptrdiff_t n = infer_size({kp, ki, kd, lower_limits, upper_limits, set_point});
        ensure_sizes(n);
        return;
    }
    if (std::holds_alternative<Mat>(value)) {
        throw std::invalid_argument("PID::set_parameter does not accept matrices for key: " + name);
    }
    throw std::invalid_argument("PID::set_parameter unsupported variant");
}

Controller::Param PIDController_1::get_parameter(const std::string& name) const {
    if (name == "kp")           return kp;
    if (name == "ki")           return ki;
    if (name == "kd")           return kd;
    if (name == "lower_limits") return lower_limits;
    if (name == "upper_limits") return upper_limits;
    if (name == "set_point")    return set_point;
    if (name == "output")       return output;
    if (name == "integrals")    return integrals;
    if (name == "prev_errors")  return prev_errors;
    throw std::invalid_argument("PID::get_parameter unknown key: " + name);
}

std::vector<std::pair<std::string,std::string>>
PIDController_1::list_parameters() const {
    return {
        {"kp",            "Per-channel proportional gains (VectorXd)"},
        {"ki",            "Per-channel integral gains (VectorXd)"},
        {"kd",            "Per-channel derivative gains (VectorXd)"},
        {"lower_limits",  "Lower clamp for integral/output (VectorXd)"},
        {"upper_limits",  "Upper clamp for integral/output (VectorXd)"},
        {"set_point",     "Desired set point (VectorXd)"},
        {"output",        "Last control output (VectorXd)"},
        {"integrals",     "Integral accumulator (VectorXd)"},
        {"prev_errors",   "Previous error (VectorXd)"},
    };
}

// =========================== LeakyIntegratorController =======================

LeakyIntegratorController::LeakyIntegratorController(const Eigen::VectorXd& K_,
                                                     const Eigen::VectorXd& alpha_,
                                                     double dt_)
    : K(K_), alpha(alpha_), dt(dt_) {
    const std::ptrdiff_t n = std::max(K.size(), alpha.size());
    ensure_size(set_point, n, 0.0);
    ensure_size(integral,  n, 0.0);
    ensure_size(output,    n, 0.0);
    ensure_size(prev_error,n, 0.0);
    ctrl_type = "LEAKY";
}

Eigen::VectorXd LeakyIntegratorController::process(const Eigen::VectorXd& measured) {
    const std::ptrdiff_t n = std::max(set_point.size(), measured.size());
    ensure_size(K, n, 0.0);
    ensure_size(alpha, n, 0.0);
    ensure_size(set_point, n, 0.0);
    ensure_size(integral, n, 0.0);
    ensure_size(output, n, 0.0);

    const Eigen::VectorXd error = measured - set_point; //set_point - measured;
    // Leaky update: integral = alpha ∘ integral + K ∘ error
    integral = alpha.cwiseProduct( (integral) + K.cwiseProduct(error)) ;
    // new_command = (previous_state + correction) * 0.99
    output = integral;
    prev_error = error;
    return output;
}

void LeakyIntegratorController::reset() {
    integral.setZero();
    output.setZero();
    prev_error.setZero();
}

void LeakyIntegratorController::set_all_gains_to_zero() {
    K.setZero();
}

void LeakyIntegratorController::set_parameter(const std::string& name, const Param& value) {
    const std::string key = leaky_norm_key(name);

    if (std::holds_alternative<Vec>(value)) {
        const auto& v = std::get<Vec>(value);

        if (key == "K")              K = v;
        else if (key == "alpha")     alpha = v;
        else if (key == "set_point") set_point = v;
        else if (key == "integral")  integral = v;
        else if (key == "output")    output = v;
        else if (key == "prev_error") prev_error = v;
        else throw std::invalid_argument("LEAKY::set_parameter unknown vector: " + name);

        const std::ptrdiff_t n = std::max(K.size(), alpha.size());
        ensure_size(set_point, n, 0.0);
        ensure_size(integral,  n, 0.0);
        ensure_size(output,    n, 0.0);
        ensure_size(prev_error,n, 0.0);
        return;
    }

    if (std::holds_alternative<Mat>(value)) {
        throw std::invalid_argument("LEAKY::set_parameter does not accept matrices for key: " + name);
    }
    throw std::invalid_argument("LEAKY::set_parameter unsupported variant");
}

Controller::Param LeakyIntegratorController::get_parameter(const std::string& name) const {
    const std::string key = leaky_norm_key(name);

    if (key == "K")          return K;
    if (key == "alpha")      return alpha;
    if (key == "set_point")  return set_point;
    if (key == "integral")   return integral;
    if (key == "output")     return output;
    if (key == "prev_error") return prev_error;

    throw std::invalid_argument("LEAKY::get_parameter unknown key: " + name);
}

// void LeakyIntegratorController::set_parameter(const std::string& name, const Param& value) {
//     if (std::holds_alternative<Vec>(value)) {
//         const auto& v = std::get<Vec>(value);
//         if (name == "K")              K = v;
//         else if (name == "alpha")     alpha = v;
//         else if (name == "set_point") set_point = v;
//         else if (name == "integral")  integral = v;
//         else if (name == "output")    output = v;
//         else if (name == "prev_error") prev_error = v;
//         else throw std::invalid_argument("LEAKY::set_parameter unknown vector: " + name);

//         const std::ptrdiff_t n = std::max(K.size(), alpha.size());
//         ensure_size(set_point, n, 0.0);
//         ensure_size(integral,  n, 0.0);
//         ensure_size(output,    n, 0.0);
//         ensure_size(prev_error,n, 0.0);
//         return;
//     }
//     if (std::holds_alternative<Mat>(value)) {
//         throw std::invalid_argument("LEAKY::set_parameter does not accept matrices for key: " + name);
//     }
//     throw std::invalid_argument("LEAKY::set_parameter unsupported variant");
// }

// Controller::Param LeakyIntegratorController::get_parameter(const std::string& name) const {
//     if (name == "K")           return K;
//     if (name == "alpha")       return alpha;
//     if (name == "set_point")   return set_point;
//     if (name == "integral")    return integral;
//     if (name == "output")      return output;
//     if (name == "prev_error")  return prev_error;
//     throw std::invalid_argument("LEAKY::get_parameter unknown key: " + name);
// }

std::vector<std::pair<std::string,std::string>>
LeakyIntegratorController::list_parameters() const {
    return {
        {"K",          "Per-channel gain (VectorXd)"},
        {"alpha",      "Leak factor in [0,1) (VectorXd)"},
        {"set_point",  "Desired set point (VectorXd)"},
        {"integral",   "Leaky integral state (VectorXd)"},
        {"output",     "Last control output (VectorXd)"},
        {"prev_error", "Previous error (VectorXd)"},
    };
}

// ================================ KalmanController ===========================

KalmanController::KalmanController(int nx, int nz, double dt_)
    : state_size(nx), measurement_size(nz), dt(dt_) {

    A = Eigen::MatrixXd::Identity(state_size, state_size);
    B = Eigen::MatrixXd::Zero(state_size, state_size);
    H = Eigen::MatrixXd::Identity(measurement_size, state_size);

    P = Eigen::MatrixXd::Identity(state_size, state_size);
    Q = 1e-6 * Eigen::MatrixXd::Identity(state_size, state_size);
    R = 1e-4 * Eigen::MatrixXd::Identity(measurement_size, measurement_size);

    x = Eigen::VectorXd::Zero(state_size);
    u = Eigen::VectorXd::Zero(state_size);
    z = Eigen::VectorXd::Zero(measurement_size);

    set_point = Eigen::VectorXd::Zero(measurement_size);
    ctrl_type = "KALMAN";
}

Eigen::VectorXd KalmanController::process(const Eigen::VectorXd& measured) {
    // Predict
    x = A * x + B * u;
    P = A * P * A.transpose() + Q;

    // Update
    z = measured;
    Eigen::MatrixXd S  = H * P * H.transpose() + R;
    Eigen::MatrixXd Kk = P * H.transpose() * S.inverse();
    x = x + Kk * (z - H * x);
    P = (Eigen::MatrixXd::Identity(P.rows(), P.cols()) - Kk * H) * P;

    // Output policy: here u = x
    u = x;
    return u;
}

void KalmanController::reset() {
    x.setZero();
    u.setZero();
    z.setZero();
    P.setIdentity();
}

void KalmanController::set_all_gains_to_zero() {
    // no-op for KF
}

void KalmanController::set_parameter(const std::string& name, const Param& value) {
    if (name == "A" && std::holds_alternative<Mat>(value)) { A = std::get<Mat>(value); return; }
    if (name == "B" && std::holds_alternative<Mat>(value)) { B = std::get<Mat>(value); return; }
    if (name == "H" && std::holds_alternative<Mat>(value)) { H = std::get<Mat>(value); return; }
    if (name == "P" && std::holds_alternative<Mat>(value)) { P = std::get<Mat>(value); return; }
    if (name == "Q" && std::holds_alternative<Mat>(value)) { Q = std::get<Mat>(value); return; }
    if (name == "R" && std::holds_alternative<Mat>(value)) { R = std::get<Mat>(value); return; }

    if (name == "x" && std::holds_alternative<Vec>(value)) { x = std::get<Vec>(value); return; }
    if (name == "u" && std::holds_alternative<Vec>(value)) { u = std::get<Vec>(value); return; }
    if (name == "z" && std::holds_alternative<Vec>(value)) { z = std::get<Vec>(value); return; }
    if (name == "set_point" && std::holds_alternative<Vec>(value)) { set_point = std::get<Vec>(value); return; }

    throw std::invalid_argument("KALMAN::set_parameter unknown key or wrong variant: " + name);
}

Controller::Param KalmanController::get_parameter(const std::string& name) const {
    if (name == "A") return A;
    if (name == "B") return B;
    if (name == "H") return H;
    if (name == "P") return P;
    if (name == "Q") return Q;
    if (name == "R") return R;

    if (name == "x") return x;
    if (name == "u") return u;
    if (name == "z") return z;
    if (name == "set_point") return set_point;

    throw std::invalid_argument("KALMAN::get_parameter unknown key: " + name);
}

std::vector<std::pair<std::string,std::string>>
KalmanController::list_parameters() const {
    return {
        {"A", "State transition (MatrixXd)"},
        {"B", "Control matrix (MatrixXd)"},
        {"H", "Measurement matrix (MatrixXd)"},
        {"P", "Estimate covariance (MatrixXd)"},
        {"Q", "Process noise covariance (MatrixXd)"},
        {"R", "Measurement noise covariance (MatrixXd)"},
        {"x", "State estimate (VectorXd)"},
        {"u", "Control/output (VectorXd)"},
        {"z", "Measurement (VectorXd)"},
        {"set_point", "Optional set point (VectorXd)"},
    };
}
