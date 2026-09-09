#include <Eigen/Dense>
#include <stdexcept>
/// To be filled out - aim to create a virtual controller that other controller types can base from
// then we can define ctrl_LO and ctrl_HO in this general way and specify/change more fluidly between different 
// types of controllers (.e.g pid, leaky, kalman)
class Controller {
public:
    virtual ~Controller() = default;
    
    // Pure virtual method for processing the measured input.
    virtual Eigen::VectorXd process(const Eigen::VectorXd& measured) = 0;
    
    // Pure virtual method for resetting the controller.
    virtual void reset() = 0;
    
    // Optionally add more common methods, e.g. set_gains(), etc.
};


// rough idea of how to implement. first in my global baldr struct (defined in baldr.h and used/configed in baldr.cpp and rtc.cpp)

// struct bdr_rtc_config {
//     // ... all other members ...
//     std::unique_ptr<Controller> ctrl_LO;
//     std::unique_ptr<Controller> ctrl_HO;
//     initDerivedParameters(){
//         // here i can define what controller type ctrl_LO and ctrl_LO will take.
//     }

//     // ...
//     // Derived runtime parameters and other members.

// void init_rtc_parameters() {
//     // Check the controller type.
//     if (rtc_config.state.controller_type == "PID") {
//         // Create PID controllers.
//         rtc_config.ctrl_LO = std::make_unique<PIDController>(rtc_config.ctrl_LO_config);
//         rtc_config.ctrl_HO = std::make_unique<PIDController>(rtc_config.ctrl_HO_config);
//     } else if (rtc_config.state.controller_type == "LEAKY") {
//         // Create a leaky integrator controller (assuming you implement one):
//         // rtc_config.ctrl_LO = std::make_unique<LeakyController>(...);
//         // rtc_config.ctrl_HO = std::make_unique<LeakyController>(...);
//         throw std::runtime_error("Leaky controller not yet implemented");
//     } else if (rtc_config.state.controller_type == "KALMAN") {
//         // Create a Kalman controller:
//         // rtc_config.ctrl_LO = std::make_unique<KalmanController>(...);
//         // rtc_config.ctrl_HO = std::make_unique<KalmanController>(...);
//         throw std::runtime_error("Kalman controller not yet implemented");
//     } else {
//         throw std::runtime_error("Unknown controller type: " + rtc_config.state.controller_type);
//     }
    
// then change (for exmaple) the pid in baldr.cpp: 

// #include "controller.h"

// class PIDController : public Controller {
// public:
//     // Existing constructors, methods, and member variables.
//     PIDController(const Eigen::VectorXd& kp_in,
//                   const Eigen::VectorXd& ki_in,
//                   const Eigen::VectorXd& kd_in,
//                   const Eigen::VectorXd& lower_limit_in,
//                   const Eigen::VectorXd& upper_limit_in,
//                   const Eigen::VectorXd& setpoint_in);
//     PIDController(const bdr_controller& config_in);
//     PIDController();
    
//     // Override the process method from Controller.
//     Eigen::VectorXd process(const Eigen::VectorXd& measured) override;
    
//     // Override the reset method from Controller.
//     void reset() override;
    
//     void set_all_gains_to_zero();

//     // Public members
//     Eigen::VectorXd kp;
//     Eigen::VectorXd ki;
//     Eigen::VectorXd kd;
//     Eigen::VectorXd lower_limits;
//     Eigen::VectorXd upper_limits;
//     Eigen::VectorXd set_point;
//     std::string ctrl_type;
//     Eigen::VectorXd output;
//     Eigen::VectorXd integrals;
//     Eigen::VectorXd prev_errors;
// };


