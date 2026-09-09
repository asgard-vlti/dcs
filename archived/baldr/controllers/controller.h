#pragma once

#include <Eigen/Dense>
#include <string>
#include <variant>
#include <vector>
#include <stdexcept>

// ----------------------------- Base Interface -------------------------------
class Controller {
public:
    using Vec   = Eigen::VectorXd;
    using Mat   = Eigen::MatrixXd;
    using Param = std::variant<Vec, Mat>;

    virtual ~Controller() = default;

    // Core RT step
    virtual Vec process(const Vec& measured) = 0;

    // Lifecycle / state mgmt
    virtual void reset() = 0;
    virtual void set_all_gains_to_zero() = 0;

    // Observability
    virtual const Vec& get_state()  const = 0;  // internal state (e.g., integral)
    virtual const Vec& get_output() const = 0;  // last control output
    virtual std::string get_type()  const = 0;

    // Setpoint
    virtual void set_setpoint(const Vec& sp) = 0;
    virtual const Vec& get_setpoint() const = 0;

    // Generic parameter surface (throw on unknown key or wrong variant)
    virtual void set_parameter(const std::string& name, const Param& value) = 0;
    virtual Param get_parameter(const std::string& name) const = 0;
    virtual std::vector<std::pair<std::string,std::string>> list_parameters() const = 0;
};

// Forward-declare to avoid including baldr.h here
struct bdr_controller;

// ============================= PIDController_1 ===============================
class PIDController_1 : public Controller {
public:
    // Public members for JSON/telemetry compatibility (Baldr-style)
    Eigen::VectorXd kp, ki, kd;                 // per-channel gains
    Eigen::VectorXd lower_limits, upper_limits; // clamp limits
    Eigen::VectorXd set_point;                  // desired value
    Eigen::VectorXd output;                     // last output
    Eigen::VectorXd integrals;                  // integral accumulator
    Eigen::VectorXd prev_errors;                // last error
    std::string     ctrl_type = "PID";

    // Ctors
    PIDController_1(); // default 140 channels
    PIDController_1(const Eigen::VectorXd& Kp,
                    const Eigen::VectorXd& Ki,
                    const Eigen::VectorXd& Kd,
                    const Eigen::VectorXd& lower,
                    const Eigen::VectorXd& upper,
                    const Eigen::VectorXd& setpoint);
    explicit PIDController_1(const bdr_controller& cfg); // (defined in .cpp if you include baldr.h)

    // Controller API
    Eigen::VectorXd process(const Eigen::VectorXd& measured) override;
    void reset() override;
    void set_all_gains_to_zero() override;

    // Observability (const refs to match base)
    const Eigen::VectorXd& get_state()  const override { return integrals; }
    const Eigen::VectorXd& get_output() const override { return output;    }
    std::string get_type() const override { return ctrl_type; }

    // Setpoint
    void set_setpoint(const Eigen::VectorXd& sp) override { set_point = sp; }
    const Eigen::VectorXd& get_setpoint() const override { return set_point; }

    // Parameters
    void set_parameter(const std::string& name, const Param& value) override;
    Param get_parameter(const std::string& name) const override;
    std::vector<std::pair<std::string,std::string>> list_parameters() const override;

private:
    void ensure_sizes(std::ptrdiff_t n);
    void clamp_vector(Eigen::VectorXd& v) const;
};

// ======================== LeakyIntegratorController ==========================
class LeakyIntegratorController : public Controller {
public:
    LeakyIntegratorController(const Eigen::VectorXd& K,
                              const Eigen::VectorXd& alpha,
                              double dt);
    ~LeakyIntegratorController() override = default;

    Eigen::VectorXd process(const Eigen::VectorXd& measured) override;
    void reset() override;
    void set_all_gains_to_zero() override;

    // Observability
    const Eigen::VectorXd& get_state()  const override { return integral; }
    const Eigen::VectorXd& get_output() const override { return output;   }
    std::string get_type() const override { return ctrl_type; }

    // Setpoint
    void set_setpoint(const Eigen::VectorXd& sp) override { set_point = sp; }
    const Eigen::VectorXd& get_setpoint() const override { return set_point; }

    // Parameters
    void set_parameter(const std::string& name, const Param& value) override;
    Param get_parameter(const std::string& name) const override;
    std::vector<std::pair<std::string,std::string>> list_parameters() const override;

private:
    Eigen::VectorXd K;         // gain
    Eigen::VectorXd alpha;     // leak factor in [0,1)
    double dt;                 // optional use in your formula

    Eigen::VectorXd set_point; // desired value
    Eigen::VectorXd integral;  // state
    Eigen::VectorXd output;    // last output
    Eigen::VectorXd prev_error;

    std::string ctrl_type = "LEAKY";
};

// ============================== KalmanController =============================
class KalmanController : public Controller {
public:
    KalmanController(int state_size, int measurement_size, double dt);
    ~KalmanController() override = default;

    Eigen::VectorXd process(const Eigen::VectorXd& measured) override;
    void reset() override;
    void set_all_gains_to_zero() override; // no-op for KF

    // Observability
    const Eigen::VectorXd& get_state()  const override { return x; }
    const Eigen::VectorXd& get_output() const override { return u; }
    std::string get_type() const override { return ctrl_type; }

    // Setpoint (optional use)
    void set_setpoint(const Eigen::VectorXd& sp) override { set_point = sp; }
    const Eigen::VectorXd& get_setpoint() const override { return set_point; }

    // Parameters
    void set_parameter(const std::string& name, const Param& value) override;
    Param get_parameter(const std::string& name) const override;
    std::vector<std::pair<std::string,std::string>> list_parameters() const override;

private:
    int state_size;
    int measurement_size;
    double dt;

    // KF matrices
    Eigen::MatrixXd A, B, H, P, Q, R;
    // KF vectors
    Eigen::VectorXd x, u, z;

    Eigen::VectorXd set_point; // optional (for external control law)

    std::string ctrl_type = "KALMAN";
};


// #pragma once

// #include <Eigen/Dense>
// #include <string>
// #include <variant>
// #include <vector>

// // ----------------------------- Base Interface -------------------------------
// class Controller {
// public:
//     using Vec   = Eigen::VectorXd;
//     using Mat   = Eigen::MatrixXd;
//     using Param = std::variant<Vec, Mat>;

//     virtual ~Controller() = default;

//     // Core RT step
//     virtual Vec process(const Vec& measured) = 0;

//     // Lifecycle / state mgmt
//     virtual void reset() = 0;
//     virtual void set_all_gains_to_zero() = 0;

//     // Observability
//     virtual const Vec& get_state()  const = 0;  // internal state (e.g., integral)
//     virtual const Vec& get_output() const = 0;  // last control output
//     virtual std::string get_type()  const = 0;

//     // Setpoint
//     virtual void set_setpoint(const Vec& sp) = 0;
//     virtual const Vec& get_setpoint() const = 0;

//     // Generic parameter surface (required; throw on unknown)
//     virtual void set_parameter(const std::string& name, const Param& value) = 0;
//     virtual Param get_parameter(const std::string& name) const = 0;
//     virtual std::vector<std::pair<std::string,std::string>> list_parameters() const = 0;
// };

// // Forward-declare to avoid including baldr.h in the header
// struct bdr_controller;

// // ============================= PIDController_1 ===============================
// class PIDController_1 : public Controller {
// public:
//     // Public members for JSON/telemetry compatibility (Baldr-style)
//     Eigen::VectorXd kp, ki, kd;                 // per-channel gains
//     Eigen::VectorXd lower_limits, upper_limits; // clamp limits
//     Eigen::VectorXd set_point;                  // desired value
//     Eigen::VectorXd output;                     // last output
//     Eigen::VectorXd integrals;                  // integral accumulator
//     Eigen::VectorXd prev_errors;                // last error
//     std::string     ctrl_type = "PID";

//     // Ctors
//     PIDController_1();
//     PIDController_1(const Eigen::VectorXd& Kp,
//                     const Eigen::VectorXd& Ki,
//                     const Eigen::VectorXd& Kd,
//                     const Eigen::VectorXd& lower,
//                     const Eigen::VectorXd& upper,
//                     const Eigen::VectorXd& setpoint);
//     explicit PIDController_1(const bdr_controller& cfg); // defined in .cpp if you include baldr.h

//     // Controller API
//     Eigen::VectorXd process(const Eigen::VectorXd& measured) override;
//     void reset() override;
//     void set_all_gains_to_zero() override;

//     // Observability (const refs to match base)
//     const Eigen::VectorXd& get_state()  const override { return integrals; }
//     const Eigen::VectorXd& get_output() const override { return output;    }
//     std::string get_type() const override { return ctrl_type; }

//     // Setpoint
//     void set_setpoint(const Eigen::VectorXd& sp) override { set_point = sp; }
//     const Eigen::VectorXd& get_setpoint() const override { return set_point; }

//     // Parameters
//     void set_parameter(const std::string& name, const Param& value) override;
//     Param get_parameter(const std::string& name) const override;
//     std::vector<std::pair<std::string,std::string>> list_parameters() const override;

// private:
//     void ensure_sizes(std::ptrdiff_t n);
//     void clamp_vector(Eigen::VectorXd& v) const;
// };

// // ======================== LeakyIntegratorController ==========================
// class LeakyIntegratorController : public Controller {
// public:
//     LeakyIntegratorController(const Eigen::VectorXd& K,
//                               const Eigen::VectorXd& alpha,
//                               double dt);
//     ~LeakyIntegratorController() override = default;

//     Eigen::VectorXd process(const Eigen::VectorXd& measured) override;
//     void reset() override;
//     void set_all_gains_to_zero() override;

//     // Observability
//     const Eigen::VectorXd& get_state()  const override { return integral; }
//     const Eigen::VectorXd& get_output() const override { return output;   }
//     std::string get_type() const override { return ctrl_type; }

//     // Setpoint
//     void set_setpoint(const Eigen::VectorXd& sp) override { set_point = sp; }
//     const Eigen::VectorXd& get_setpoint() const override { return set_point; }

//     // Parameters
//     void set_parameter(const std::string& name, const Param& value) override;
//     Param get_parameter(const std::string& name) const override;
//     std::vector<std::pair<std::string,std::string>> list_parameters() const override;

// private:
//     Eigen::VectorXd K;         // gain
//     Eigen::VectorXd alpha;     // leak factor in [0,1)
//     double dt;                 // optional use in your formula

//     Eigen::VectorXd set_point; // desired value
//     Eigen::VectorXd integral;  // state
//     Eigen::VectorXd output;    // last output
//     Eigen::VectorXd prev_error;

//     std::string ctrl_type = "LEAKY";
// };

// // ============================== KalmanController =============================
// class KalmanController : public Controller {
// public:
//     KalmanController(int state_size, int measurement_size, double dt);
//     ~KalmanController() override = default;

//     Eigen::VectorXd process(const Eigen::VectorXd& measured) override;
//     void reset() override;
//     void set_all_gains_to_zero() override; // no-op for KF

//     // Observability
//     const Eigen::VectorXd& get_state()  const override { return x; }
//     const Eigen::VectorXd& get_output() const override { return u; }
//     std::string get_type() const override { return ctrl_type; }

//     // Setpoint (optional use)
//     void set_setpoint(const Eigen::VectorXd& sp) override { set_point = sp; }
//     const Eigen::VectorXd& get_setpoint() const override { return set_point; }

//     // Parameters
//     void set_parameter(const std::string& name, const Param& value) override;
//     Param get_parameter(const std::string& name) const override;
//     std::vector<std::pair<std::string,std::string>> list_parameters() const override;

// private:
//     int state_size;
//     int measurement_size;
//     double dt;

//     // KF matrices
//     Eigen::MatrixXd A, B, H, P, Q, R;
//     // KF vectors
//     Eigen::VectorXd x, u, z;

//     Eigen::VectorXd set_point; // optional (for external control law)

//     std::string ctrl_type = "KALMAN";
// };
