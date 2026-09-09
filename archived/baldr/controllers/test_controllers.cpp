// #define CATCH_CONFIG_MAIN
// #include "catch.hpp"
// #include "controller.h"
// #include <Eigen/Dense>

// from this directory (on mac)
//clang++ -std=c++17 -Wall -I/opt/homebrew/opt/eigen/include/eigen3 -I../../catch2 test_controllers.cpp controller.cpp -o test_controllers
// or on ubuntu: g++ -std=c++17 -Wall -I/usr/include/eigen3 -I../../catch2 test_controllers.cpp controller.cpp -o test_controllers

// test_controllers.cpp

#define CATCH_CONFIG_MAIN
#include "controller.h"
#include <cmath>
#include "catch.hpp"
#include "controller.h"
#include <Eigen/Dense>


// ---------- helpers ----------
static inline Eigen::VectorXd V(std::initializer_list<double> xs) {
    Eigen::VectorXd v(xs.size());
    int i = 0; for (double x : xs) v[i++] = x;
    return v;
}

static inline void approx_vec(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double eps=1e-9) {
    REQUIRE(a.size() == b.size());
    for (Eigen::Index i = 0; i < a.size(); ++i) {
        INFO("i=" << i << " a=" << a[i] << " b=" << b[i]);
        REQUIRE(std::abs(a[i] - b[i]) <= eps);
    }
}

// ============================================================================
//                                   PID
// ============================================================================
TEST_CASE("PIDController_1: digital step, anti-windup, soft clamp", "[pid]") {
    // 3 channels
    const Eigen::VectorXd kp = V({1.0, 2.0, 3.0});
    const Eigen::VectorXd ki = V({0.5, 0.0, 0.1});
    const Eigen::VectorXd kd = V({0.2, 0.3, 0.0});

    // Tight limits to exercise clamps
    const Eigen::VectorXd lo = V({-0.5, -0.2, -0.05});
    const Eigen::VectorXd hi = V({ +0.5, +0.2, +0.05});

    Eigen::VectorXd sp = V({1.0, -1.0, 0.1});
    PIDController_1 pid(kp, ki, kd, lo, hi, sp);

    Eigen::VectorXd y = V({0.0, 0.0, 0.0}); // measured

    // Step 1 math (digital):
    // e=[1,-1,0.1], I+=Ki∘e, D=Kd∘(e-0), P=Kp∘e, clamp output to [lo,hi]
    Eigen::VectorXd u1 = pid.process(y);
    approx_vec(u1, V({0.5, -0.2, 0.05}));

    // Step 2: same error; D=0; integral accumulates and clamps; output stays clamped
    Eigen::VectorXd u2 = pid.process(y);
    approx_vec(u2, V({0.5, -0.2, 0.05}));

    // State exposure
    const auto integ = std::get<Eigen::VectorXd>(pid.get_parameter("integrals"));
    const auto prev  = std::get<Eigen::VectorXd>(pid.get_parameter("prev_errors"));
    REQUIRE(integ.size() == 3);
    REQUIRE(prev.size() == 3);
    REQUIRE(integ[0] == Approx(0.5));
    REQUIRE(integ[1] == Approx(0.0));
    REQUIRE(integ[2] == Approx(0.02).epsilon(1e-12));
    approx_vec(prev, V({1.0, -1.0, 0.1}));

    // Zero gains: output equals (clamped) integrals
    pid.set_all_gains_to_zero();
    Eigen::VectorXd u3 = pid.process(y);
    approx_vec(u3, integ);
}

TEST_CASE("PIDController_1: parameter API and setpoint wiring (consistent size)", "[pid]") {
    PIDController_1 pid(
        V({1,2,3,4}),          // kp
        V({0,0,0,0}),          // ki
        V({0,0,0,0}),          // kd
        V({-1,-1,-1,-1}),      // lower_limits
        V({+1,+1,+1,+1}),      // upper_limits
        V({1,1,1,1})           // set_point
    );

    // Make sure output won’t clamp for this proportional-only check
    pid.set_parameter("lower_limits", V({-10, -10, -10, -10}));
    pid.set_parameter("upper_limits", V({+10, +10, +10, +10}));

    // Now mutate via parameter API (still size 4)
    pid.set_parameter("kp", V({1,2,3,4}));
    pid.set_parameter("ki", V({0,0,0,0}));
    pid.set_parameter("kd", V({0,0,0,0}));

    Eigen::VectorXd m = V({0,0,0,0});
    Eigen::VectorXd u = pid.process(m);
    approx_vec(u, V({1,2,3,4})); // pure proportional, no clamping now

    auto got_kp = std::get<Eigen::VectorXd>(pid.get_parameter("kp"));
    approx_vec(got_kp, V({1,2,3,4}));
    REQUIRE(pid.get_type() == "PID");
}


// ============================================================================
//                               Leaky Integrator
// ============================================================================
TEST_CASE("LeakyIntegratorController: leaky accumulation", "[leaky]") {
    const Eigen::VectorXd K = V({0.2, 0.5});
    const Eigen::VectorXd A = V({0.9, 0.8}); // alpha
    LeakyIntegratorController L(K, A, /*dt=*/1.0);

    L.set_setpoint(V({1.0, -1.0}));
    Eigen::VectorXd y = V({0.0, 0.0});

    // step1: integral = A∘0 + K∘e = [0.2, -0.5]
    Eigen::VectorXd u1 = L.process(y);
    approx_vec(u1, V({0.2, -0.5}));

    // step2: integral = A∘[0.2,-0.5] + K∘e = [0.38, -0.9]
    Eigen::VectorXd u2 = L.process(y);
    approx_vec(u2, V({0.38, -0.9}));

    L.set_all_gains_to_zero();
    Eigen::VectorXd u3 = L.process(y); // pure leak
    REQUIRE(u3[0] == Approx(0.9 * 0.38));
    REQUIRE(u3[1] == Approx(0.8 * -0.9));
    L.reset();
    approx_vec(L.get_state(), V({0.0, 0.0}));
}

// ============================================================================
//                                   Kalman
// ============================================================================
TEST_CASE("KalmanController: dimensions and update sanity", "[kalman]") {
    const int nx = 3, nz = 3;
    KalmanController KF(nx, nz, /*dt=*/1.0);
    KF.set_setpoint(Eigen::VectorXd::Zero(nz)); // optional

    Eigen::VectorXd z = V({1.0, -2.0, 0.5});
    Eigen::VectorXd u = KF.process(z);

    REQUIRE(u.size() == nx);
    for (int i = 0; i < nx; ++i) REQUIRE(std::isfinite(u[i]));

    // Matrix parameter roundtrip
    Eigen::MatrixXd Anew = 2.0 * Eigen::MatrixXd::Identity(nx, nx);
    KF.set_parameter("A", Anew);
    auto Aret = std::get<Eigen::MatrixXd>(KF.get_parameter("A"));
    REQUIRE((Aret - Anew).norm() == Approx(0.0));
}


// // ---------- small helpers ----------
// static inline Eigen::VectorXd V(std::initializer_list<double> xs) {
//     Eigen::VectorXd v(xs.size());
//     int i = 0; for (double x : xs) v[i++] = x;
//     return v;
// }

// static inline void approx_vec(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double eps=1e-9) {
//     REQUIRE(a.size() == b.size());
//     for (Eigen::Index i = 0; i < a.size(); ++i) {
//         INFO("i=" << i << " a=" << a[i] << " b=" << b[i]);
//         REQUIRE(std::abs(a[i] - b[i]) <= eps);
//     }
// }

// // ============================================================================
// //                                   PID
// // ============================================================================
// TEST_CASE("PIDController_1: digital step, anti-windup, soft clamp", "[pid]") {
//     // 3 channels
//     const Eigen::VectorXd kp = V({1.0, 2.0, 3.0});
//     const Eigen::VectorXd ki = V({0.5, 0.0, 0.1});
//     const Eigen::VectorXd kd = V({0.2, 0.3, 0.0});

//     // Tight limits to exercise clamps
//     const Eigen::VectorXd lo = V({-0.5, -0.2, -0.05});
//     const Eigen::VectorXd hi = V({ +0.5, +0.2, +0.05});

//     Eigen::VectorXd sp = V({1.0, -1.0, 0.1});
//     PIDController_1 pid(kp, ki, kd, lo, hi, sp);

//     Eigen::VectorXd y = V({0.0, 0.0, 0.0}); // measured

//     // Step 1 math (digital):
//     // e = [1, -1, 0.1]
//     // I += Ki∘e = [0.5, 0, 0.01]  (no clamp yet)
//     // D = Kd∘(e-0) = [0.2, -0.3, 0]
//     // P = Kp∘e = [1, -2, 0.3]
//     // u_raw = [1.7, -2.3, 0.31]  -> clamp to [0.5, -0.2, 0.05]
//     Eigen::VectorXd u1 = pid.process(y);
//     approx_vec(u1, V({0.5, -0.2, 0.05}));

//     // Step 2: same error; D=0 now; integral accumulates and clamps
//     Eigen::VectorXd u2 = pid.process(y);
//     approx_vec(u2, V({0.5, -0.2, 0.05}));

//     // State exposure
//     const auto integ = std::get<Eigen::VectorXd>(pid.get_parameter("integrals"));
//     const auto prev  = std::get<Eigen::VectorXd>(pid.get_parameter("prev_errors"));
//     REQUIRE(integ.size() == 3);
//     REQUIRE(prev.size() == 3);
//     // ch0 integral should be clamped to +0.5, ch1 0, ch2 small
//     REQUIRE(integ[0] == Approx(0.5));
//     REQUIRE(integ[1] == Approx(0.0));
//     REQUIRE(integ[2] == Approx(0.02).epsilon(1e-12));
//     approx_vec(prev, V({1.0, -1.0, 0.1}));

//     // Zero gains: now output should stay equal to (clamped) integrals (Ki=0 so no further growth)
//     pid.set_all_gains_to_zero();
//     Eigen::VectorXd u3 = pid.process(y);
//     approx_vec(u3, integ);
// }

// TEST_CASE("PIDController_1: parameter API and setpoint wiring", "[pid]") {
//     PIDController_1 pid; // default 140 channels
//     // Resize & set via parameter API
//     pid.set_parameter("kp", V({1,2,3,4}));
//     pid.set_parameter("ki", V({0,0,0,0}));
//     pid.set_parameter("kd", V({0,0,0,0}));
//     pid.set_parameter("lower_limits", V({-1,-1,-1,-1}));
//     pid.set_parameter("upper_limits", V({+1,+1,+1,+1}));
//     pid.set_parameter("set_point", V({1,1,1,1}));

//     Eigen::VectorXd m = V({0,0,0,0});
//     Eigen::VectorXd u = pid.process(m);
//     approx_vec(u, V({1,2,3,4})); // pure proportional

//     auto got_kp = std::get<Eigen::VectorXd>(pid.get_parameter("kp"));
//     approx_vec(got_kp, V({1,2,3,4}));
//     REQUIRE(pid.get_type() == "PID");
// }

// // ============================================================================
// //                               Leaky Integrator
// // ============================================================================
// TEST_CASE("LeakyIntegratorController: leaky accumulation", "[leaky]") {
//     const Eigen::VectorXd K = V({0.2, 0.5});
//     const Eigen::VectorXd A = V({0.9, 0.8}); // alpha
//     LeakyIntegratorController L(K, A, /*dt=*/1.0);

//     L.set_setpoint(V({1.0, -1.0}));
//     Eigen::VectorXd y = V({0.0, 0.0});

//     // step1: integral = A∘0 + K∘e = [0.2, -0.5]
//     Eigen::VectorXd u1 = L.process(y);
//     approx_vec(u1, V({0.2, -0.5}));

//     // step2: integral = A∘[0.2,-0.5] + K∘e = [0.38, -0.9]
//     Eigen::VectorXd u2 = L.process(y);
//     approx_vec(u2, V({0.38, -0.9}));

//     L.set_all_gains_to_zero();
//     Eigen::VectorXd u3 = L.process(y); // pure leak
//     REQUIRE(u3[0] == Approx(0.9 * 0.38));
//     REQUIRE(u3[1] == Approx(0.8 * -0.9));
//     L.reset();
//     approx_vec(L.get_state(), V({0.0, 0.0}));
// }

// // ============================================================================
// //                                   Kalman
// // ============================================================================
// TEST_CASE("KalmanController: dimensions and update sanity", "[kalman]") {
//     const int nx = 3, nz = 3;
//     KalmanController KF(nx, nz, /*dt=*/1.0);
//     KF.set_setpoint(Eigen::VectorXd::Zero(nz)); // optional, not used in simple policy

//     Eigen::VectorXd z = V({1.0, -2.0, 0.5});
//     Eigen::VectorXd u = KF.process(z);

//     REQUIRE(u.size() == nx);
//     for (int i = 0; i < nx; ++i) REQUIRE(std::isfinite(u[i]));

//     // Matrix parameter roundtrip
//     Eigen::MatrixXd Anew = 2.0 * Eigen::MatrixXd::Identity(nx, nx);
//     KF.set_parameter("A", Anew);
//     auto Aret = std::get<Eigen::MatrixXd>(KF.get_parameter("A"));
//     REQUIRE((Aret - Anew).norm() == Approx(0.0));
// }



