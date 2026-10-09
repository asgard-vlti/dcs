#include "heimdallr.h"
#include "ddspc_snapshot.hpp"
#include "ddspc_piston_hold.hpp"
#include "fringe_frame_wait.hpp"
#include "predictive_control.hpp"
#include <chrono>
#include <random>
//#define PRINT_TIMING
//#define PRINT_TIMING_ALL
//#define DEBUG
//#define DEBUG_FILTER6

#define MAX_DM_PISTON 0.4
// Group delay is in wavelengths at 2.05 microns. Need 0.5 waves to be 2.5 sigma.
#define GD_MAX_VAR_FOR_JUMP 0.2*0.2
#define GD_MIN_REAL_VAR 1E-6
#define N_MOD 6 //14
#define MODULATION_AMPLITUDE 25.0 

using namespace std::complex_literals;

long unsigned int ft_cnt=0, cnt_since_init=0;
int mod_ix=0, gd_ix=0; //!!!This shouldn't be a local global.
long unsigned int nerrors=0;
double gd_to_K1=1.0;

// Local baseline variables
dcomp K1_phasor[N_BL], K2_phasor[N_BL];

// A 6x6 matrix for the weights of phase and group delay
Eigen::Matrix<double, N_BL, 1> var_pd, var_gd, Wpd, Wgd;
Eigen::Matrix<double, N_BL, 1> pd_filtered, gd_filtered;

// Convenience matrices and vectors.
// A 4x4 matrix of zeros to store the diagonal.
Eigen::Matrix<double, N_TEL, N_TEL> singularDiag = Eigen::Matrix<double, N_TEL, N_TEL>::Zero();

// A 4x4 identity matrix.
Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

// A constant N_TEL x N_MOD matrix for modulation, and an equivalent for baselines.
/*Eigen::Matrix<double, N_TEL, N_MOD> modulation_matrix = (Eigen::Matrix<double, N_TEL, N_MOD>() <<
    1,0,0,1,1,0,0,0,1,0,1,1,1,0,
    0,1,0,1,1,0,0,1,0,1,0,1,0,1,
    0,0,1,0,1,0,1,0,0,1,1,0,1,1,    
    1,1,1,1,0,1,0,0,0,1,1,0,0,0).finished();*/
Eigen::Matrix<double, N_TEL, N_MOD> modulation_matrix = (Eigen::Matrix<double, N_TEL, N_MOD>() <<
    1,0,0,1,0,1,
    0,1,1,0,0,1,
    0,1,0,1,0,1,   
    0,1,0,1,1,0).finished();
double bmodn[6] = {2,1,2,1,2,1};
double bzn[6] = {2,4,2,4,2,4}; 

Eigen::Matrix<double, N_BL, N_MOD> bl_modulation_matrix = M_lacour * modulation_matrix;
// Saving SNR during the modulation.
Eigen::Matrix<double, N_BL, N_MOD> gd_snr_during_mod = Eigen::Matrix<double, N_BL, N_MOD>::Zero();

// The search vector. There is no reason for this to have 
// diferent frequencies for each baseline.
//Eigen::Vector4d search_vector_scale(-2.75,-1.75,1.25,3.25);
Eigen::Vector4d search_vector_scale(-1.5,-0.5,0.5,1.5);

template <typename T> int sgn(T val){
	return (T(0) < val) - (val < T(0));
}

// Make the pseudo-inverse matrix needed to project onto delay line (telescope) space.
// W is the diagonal of a 6x6 matrix of weights for each baseline.
#define NUMERIC_LIMIT 2e-6
Eigen::Matrix4d make_pinv(Eigen::Matrix<double, N_BL, 1> W, double threshold){
    using namespace Eigen;
    // This function computes the pseudo-inverse of the matrix M^T *  W * M, using the
    // SVD method. The threshold is used to set the minimum eigenvalue, and the
    // minimum eigenvalue is used to set the minimum eigenvalue of the pseudo-inverse.
    // W * M_lacour is a 6x4 matrix, and M_lacour.transpose() * W * M_lacour is a 4x4 matrix.
#ifdef PRINT_TIMING_ALL
    timespec now, then;
    clock_gettime(CLOCK_REALTIME, &then);
#endif
    SelfAdjointEigenSolver<Matrix4d> es(M_lacour.transpose() * W.asDiagonal() * M_lacour);
#ifdef PRINT_TIMING_ALL
    clock_gettime(CLOCK_REALTIME, &now);
    if (then.tv_sec == now.tv_sec)
        info("SVD time: %ld", now.tv_nsec - then.tv_nsec);
    then = now;
#endif
    // Start with a diagonal vector of 4 zeros.
    for (int i=0; i<N_TEL; i++){
         if ((es.eigenvalues()(i) <= threshold) || (es.eigenvalues()(i) < NUMERIC_LIMIT)){
             if (threshold > 0){
                 singularDiag(i,i) = es.eigenvalues()(i)/threshold/threshold;
            } else singularDiag(i,i) = 0;
        } else {
            singularDiag(i,i) = 1.0/es.eigenvalues()(i);
        }
    }
#ifdef PRINT_TIMING_ALL
    clock_gettime(CLOCK_REALTIME, &now);
    if (then.tv_sec == now.tv_sec)
        info("Thresholding time: %ld", now.tv_nsec - then.tv_nsec);
#endif
    return  es.eigenvectors() * singularDiag * es.eigenvectors().transpose();
}

// Normalized sinc function
double sinc_normalized(double x) {
    if (x == 0.0) {
        return 1.0;
    } else {
        return std::sin(M_PI * x) / (M_PI * x);
    }
}

void start_modulation() {
    // This function starts the modulation by setting the first modulation pattern.
    while(gd_ix != 0){
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    mod_ix = 0;
    set_mod(MODULATION_AMPLITUDE * modulation_matrix.col(mod_ix));
    sem_post(&sem_offload);
}

void end_modulation() {
    // This function starts the modulation by setting the first modulation pattern.
    mod_ix = 0;
    set_mod(Eigen::Vector4d::Zero());
    sem_post(&sem_offload);
}

void set_dm_piston(Eigen::Vector4d dm_piston, bool force_write = false){
#ifdef SIMULATE
    (void)force_write;
    static IMAGE piston_cmd = {};
    static bool piston_cmd_open = false;
    static bool piston_cmd_error_logged = false;
    if (!piston_cmd_open) {
        if (ImageStreamIO_openIm(&piston_cmd, "piston_cmd") != IMAGESTREAMIO_SUCCESS) {
            if (!piston_cmd_error_logged) {
                error("Failed to open simulator piston_cmd image stream.");
                piston_cmd_error_logged = true;
            }
            return;
        }
        if (piston_cmd.md->naxis != 2 ||
            piston_cmd.md->size[0] * piston_cmd.md->size[1] != N_TEL ||
            piston_cmd.md->datatype != _DATATYPE_FLOAT) {
            error("Simulator piston_cmd must be a 4-element float32 image stream.");
            ImageStreamIO_closeIm(&piston_cmd);
            return;
        }
        piston_cmd_open = true;
    }

    piston_cmd.md->write = 1;
    for (int i = 0; i < N_TEL; i++) {
        // The simulator input is OPD in metres; DM piston units are microns of OPD.
        piston_cmd.array.F[i] = static_cast<float>(dm_piston(i) * OPD_PER_DM_UNIT * 1e-6);
    }
    piston_cmd.md->write = 0;
    piston_cmd.md->cnt0++;
    ImageStreamIO_sempost(&piston_cmd, -1);
#else
    // Make sure that we only move the DM for for the active beams.
    control_u.dm_piston = control_u.beams_active.asDiagonal() * control_u.dm_piston;       	
    // This function sets the DM piston to the given value.
    for(int i = 0; i < N_TEL; i++) {
        if (control_u.search(i) != 0.0 && !force_write) {
            control_u.dm_piston(i) = 0.0; // Reset DM piston if in search mode
            continue; // Do not set if in search mode
        }
        for (int j=0; j<144; j++){
            DMs[i].array.D[j] = dm_piston(i);
        }
        ImageStreamIO_sempost(&master_DMs[i], 1);
    }
#endif
}

// Initialise variables assocated with baselines, including 
// bispectra.
void initialise_baselines(){
    debug("BL Modulation matrix:\n%s", log_stringify(bl_modulation_matrix).c_str());
    cnt_since_init = 0;
    Wpd.setZero();
    Wgd.setZero();
    var_gd.setZero();
    var_pd.setZero();
    pd_filtered.setZero();
    gd_filtered.setZero();
    baselines.gd.setZero();
    baselines.pd.setZero();
    baselines.gd_snr.setZero();
    baselines.pd_snr.setZero();
    // This also sets to zero.
    baselines.set_gd_boxcar(INIT_N_GD_BOXCAR);
    baselines.n_pd_boxcar=MAX_N_PD_BOXCAR;
    baselines.pd_phasor.setZero();
    baselines.pd_phasor_boxcar_avg.setZero();
    baselines.pd_av_filtered.setZero();
    baselines.pd_av.setZero();

    for (unsigned int i=0; i<baselines.n_pd_boxcar; i++){
        baselines.pd_phasor_boxcar[i].setZero();
    }

    // Reset bispectra variables for K1 and K2
    for (int i=0; i<N_CP; i++){
        bispectra_K1[i].n_bs_boxcar=MAX_N_BS_BOXCAR;
        bispectra_K1[i].ix_bs_boxcar=0;
        bispectra_K1[i].bs_phasor = 0;
        bispectra_K1[i].closure_phase = 0;
        for (int j=0; j<MAX_N_BS_BOXCAR; j++){
            bispectra_K1[i].bs_phasors[j] = 0;
        }
        bispectra_K2[i].n_bs_boxcar=MAX_N_BS_BOXCAR;
        bispectra_K2[i].ix_bs_boxcar=0;
        bispectra_K2[i].bs_phasor = 0;
        bispectra_K2[i].closure_phase = 0;
        for (int j=0; j<MAX_N_BS_BOXCAR; j++){
            bispectra_K2[i].bs_phasors[j] = 0;
        }
    }
    float wave_K1 = config["wave"]["K1"].value_or(2.05);
    float wave_K2 = config["wave"]["K2"].value_or(2.25);
    gd_to_K1 = wave_K2/(wave_K2-wave_K1)/2/M_PI;

    for (int bl=0; bl<N_BL; bl++){
        // Set the offsets to the group delay
#ifndef SIMULATE
        baselines.gd_phasor_offset(bl) = 
            std::exp(-1.0i *config["servo"]["gd_phasor_offset"][bl].value_or(0.0)/gd_to_K1);
#else
        baselines.gd_phasor_offset(bl) = 
            std::exp(-1.0i *config["servo"]["gd_phasor_sim_offset"][bl].value_or(0.0)/gd_to_K1);
#endif
    }
}

// Reset the search
void reset_search(){
    // This function resets the search for the delay line and piezo.
    // It sets the delay line and piezo to zero, and sets the SNR values to zero.
    beam_mutex.lock();
    control_u.dl.setZero();
    control_u.piezo.setZero();
    control_u.dm_piston.setZero();
    control_u.search.setZero();
    control_u.dl_offload.setZero();
    // In microns. If running at 250Hz with 50Hz offloading, 
    // we can't move more than a fraction of a  
    // coherence length in 32 samples=~6 offloads.
    control_u.search_delta = 1.0; 
    control_u.steps_to_turnaround = 10;
    control_u.search_Nsteps = 0;
    control_u.dit = 0.001; // Default to 1ms
    control_u.test_beam=0;
    control_u.test_n=0;
    control_u.test_ix=0;
    control_u.test_value=0.1;
    control_u.fringe_found = false;
    control_u.itime=0;
    beam_mutex.unlock();
}

// This takes 12 microseconds on average. A little slow!
Eigen::Matrix<double, N_BL, 1> filter6(Eigen::Matrix<double, N_BL, N_BL> I6, Eigen::Matrix<double, N_BL, 1> x, Eigen::Matrix<double, N_BL, 1> W){
    // This function filters the input vector x using the I6gd matrix.
    // It returns the filtered vector.
    double chi2=1e6, chi2_min=1e6;
 //   int i_best;
    Eigen::Matrix<double, N_BL, 1> y_best, x_try, x_best;
    Eigen::Matrix<double, N_BL, 1> y;
    // Each positive element of x could be x-1, and each negative element
    // could be x+1. If we tried every combination, we would have 2^N_BL=64 combinations.
    // The best combination has the minimum chi^2 of the modified x with 
    // respect to the final y

    // For debugging, input a very simple I6 matrix, applicable to infinite SNR.
    // Instead of using << and .finished(), use a static array and Eigen::Map:
    /*static const double I6_data[N_BL*N_BL] = {
        0.5, 0.25, 0.25, -0.25, -0.25, 0.0,
        0.25, 0.5, 0.25, 0.25, 0.0, -0.25,
        0.25, 0.25, 0.5, 0.0, 0.25, 0.25,
        -0.25, 0.25, 0.0, 0.5, 0.25, -0.25,
        -0.25, 0.0, 0.25, 0.25, 0.5, 0.25,
        0.0, -0.25, 0.25, -0.25, 0.25, 0.5
    };
    I6 = Eigen::Map<const Eigen::Matrix<double, N_BL, N_BL>>(I6_data); */
    for (unsigned int i=0; i<(1<<N_BL); i++){
    	// We only change up to 2 baselines here. 
    	if (__builtin_popcount(i) > 2) continue;
        for (unsigned int j=0; j<N_BL; j++){
            if (x(j) > 0){
                if (i & (1<<j)){
                    x_try(j) = x(j) - 1.0;
                } else {
                    x_try(j) = x(j);
                }
            } else {
                if (i & (1<<j)){
                    x_try(j) = x(j) + 1.0;
                } else {
                    x_try(j) = x(j);
                }
            }
        }
        y = I6 * x_try;
        chi2 = (x_try - y).transpose() * W.asDiagonal() * (x_try - y);
        if (chi2 < chi2_min){
            chi2_min = chi2;
            y_best = y;
            x_best = x_try;
//            i_best = i;
        }
    }
#ifdef DEBUG_FILTER6
    // For debugging, print the best combination found, x_best, and y_best
    info("%s", fmt::format("Best i {:b}", i_best).c_str());
    // Print Eigen vectors as comma-separated values
    {
        std::ostringstream stream;
        stream << "Initial x:       ";
        for (int k = 0; k < x.size(); ++k) {
            stream << fmt::format("{:.4f}", x(k)) << ((k < x.size()-1) ? ", " : "");
        }
        info("%s", stream.str().c_str());
    }
    {
        std::ostringstream stream;
        stream << "Best modified x: ";
        for (int k = 0; k < x_best.size(); ++k) {
            stream << fmt::format("{:.4f}", x_best(k)) << ((k < x_best.size()-1) ? ", " : "");
        }
        info("%s", stream.str().c_str());
    }
    {
        std::ostringstream stream;
        stream << "Best y:           ";
        for (int k = 0; k < y_best.size(); ++k) {
            stream << fmt::format("{:.4f}", y_best(k)) << ((k < y_best.size()-1) ? ", " : "");
        }
        info("%s", stream.str().c_str());
    }
    //y_best = I6 * x;
#endif
    chi2 = (x - y).transpose() * W.asDiagonal() * (x - y);
    return y_best;
}

// The main fringe tracking function
void fringe_tracker(){
    timespec now, last_dl_offload;
#ifdef DEBUG_ALL
    timespec now_all, then_all;
#endif
    last_dl_offload.tv_sec = 0;
    last_dl_offload.tv_nsec = 0;
    using namespace std::complex_literals;
    Eigen::Matrix<double, N_BL, N_BL> I6gd, I6pd;
    Eigen::Matrix4d I4_search_projection;
    Eigen::Matrix<double, N_TEL, N_TEL> cov_gd_tel;
    Eigen::Matrix<double, N_TEL, N_TEL> cov_pd_tel;
    Eigen::Vector4d pd_gain_scale = Eigen::Vector4d::Ones();
    unsigned long int last_gd_jump=0;
    heimdallr_ddspc::DdspcServo ddspc;
    heimdallr_ddspc::PistonResetHold piston_reset_hold;
    heimdallr_ddspc::Parameters ddspc_params;
    std::mt19937_64 exploration_generator(std::random_device{}());
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    std::uint64_t seen_servo_transition = 0;
#ifdef SIMULATE
    bool camera_counter_offset_set = false;
    long unsigned int k1_counter_origin = 0;
    long unsigned int k2_counter_origin = 0;
#endif
    auto next_ddspc_wait_log = std::chrono::steady_clock::time_point{};
    auto next_ddspc_active_log = std::chrono::steady_clock::time_point{};
    auto next_frame_gap_log = std::chrono::steady_clock::time_point{};
    unsigned long int frame_gap_streak = 0;
    unsigned long int frame_gaps_since_log = 0;
    auto report_ddspc_wait = [&](const char *reason) {
        {
            std::lock_guard<std::mutex> lock(settings.mutex);
            if (settings.s.servo_mode != SERVO_DDSPC) return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now < next_ddspc_wait_log) return;
        info("DDSPC waiting: %s", reason);
        next_ddspc_wait_log = now + std::chrono::seconds(1);
    };
    auto reset_ddspc_fit = [&](const char *reason, bool early_exit) {
        bool selected;
        {
            std::lock_guard<std::mutex> lock(settings.mutex);
            selected = settings.s.servo_mode == SERVO_DDSPC;
        }
        const int updates = ddspc.controller().iterations();
        const bool reset_fit = selected && !ddspc.frozen() && updates > 0;
        ddspc.invalidate();
        if (reset_fit) {
            info("DDSPC paused: %s; resetting fit after %d valid updates",
                 reason, updates);
            piston_reset_hold.arm();
        } else {
            report_ddspc_wait(reason);
        }
        if (reset_fit || (early_exit && selected && piston_reset_hold.active())) {
            control_u.dm_piston = piston_reset_hold.command(
                control_u.dm_piston, true);
            set_dm_piston(control_u.dm_piston, true);
        }
    };
    auto process_servo_transitions = [&] {
        if (seen_servo_transition ==
            settings.servo_transition_generation.load(std::memory_order_acquire))
            return;
        std::deque<heimdallr_ddspc::ServoTransition> transitions;
        {
            std::lock_guard<std::mutex> lock(settings.mutex);
            transitions.swap(settings.servo_transitions);
            seen_servo_transition = settings.servo_transition_generation.load(
                std::memory_order_relaxed);
        }
        for (const auto& transition : transitions) {
            if (transition.freeze) {
                ddspc.freeze_manual();
                {
                    std::lock_guard<std::mutex> lock(settings.mutex);
                    settings.ddspc_freeze_status.pending = false;
                    settings.ddspc_freeze_status.frozen = ddspc.frozen();
                    settings.ddspc_freeze_status.reason = ddspc.freeze_reason();
                }
                info("DDSPC learning frozen by command after %d valid frames",
                     ddspc.exploration_frames());
                continue;
            }
            if (transition.from == SERVO_DDSPC) {
                piston_reset_hold.clear();
                if (transition.to == SERVO_OFF) {
                    try {
                        heimdallr_ddspc::SnapshotJob job;
                        job.model = ddspc.snapshot_for_off();
                        job.transition_time_ns = transition.time_ns;
                        job.sequence = transition.sequence;
                        job.trigger = transition.trigger;
                        if (!ddspc_snapshot_writer->enqueue(std::move(job))) {
                            error("DDSPC snapshot queue is stopped");
                        }
                    } catch (const std::exception& e) {
                        error("DDSPC snapshot capture failed: %s", e.what());
                        ddspc.invalidate();
                    }
                } else {
                    ddspc.invalidate();
                }
                std::lock_guard<std::mutex> lock(settings.mutex);
                settings.ddspc_active_valid = false;
                settings.ddspc_freeze_status = {};
            }
            if (transition.to == SERVO_DDSPC) {
                piston_reset_hold.clear();
                ddspc_params = transition.ddspc_params;
                ddspc.enter(ddspc_params);
                {
                    std::lock_guard<std::mutex> lock(settings.mutex);
                    settings.ddspc_active = ddspc_params;
                    settings.ddspc_active_valid = true;
                    settings.ddspc_freeze_status =
                        {false, ddspc.frozen(), ddspc.freeze_reason()};
                }
                next_ddspc_wait_log = std::chrono::steady_clock::time_point{};
                next_ddspc_active_log = std::chrono::steady_clock::time_point{};
                info("DDSPC selected; waiting for valid four-beam tracking");
            }
        }
    };

    long x_px, y_px, stride;
    initialise_baselines();
    reset_search();
    set_dm_piston(Eigen::Vector4d::Zero()); 
    ft_cnt = K1ft->cnt;
    bool k1_ready = false;
    bool k2_ready = false;
    while (true) {
        process_servo_transitions();
        {
            std::lock_guard<std::mutex> lock(settings.mutex);
            if (settings.s.servo_mode == SERVO_STOP) break;
        }
        auto wait_for_frame = [&](sem_t* semaphore, bool& ready) {
            try {
                heimdallr_ddspc::wait_for_frame(semaphore, ready);
            } catch (const std::system_error& e) {
                warn("%s", e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };
        wait_for_frame(&K1ft->sem_new_frame, k1_ready);
        process_servo_transitions();
        if (!k1_ready) continue;
        wait_for_frame(&K2ft->sem_new_frame, k2_ready);
        process_servo_transitions();
        if (!k2_ready) continue;
        k1_ready = false;
        k2_ready = false;
        {
            std::lock_guard<std::mutex> lock(settings.mutex);
            if (settings.s.servo_mode == SERVO_STOP) break;
        }
        bool ddspc_frame_gap = false;
        cnt_since_init++; //This should "never" wrap around, as a long int is big.
        if ((K1ft->bad_frame) || (K2ft->bad_frame)) {
            reset_ddspc_fit("bad camera frame", true);
            ft_cnt++;
            continue;
        }
        const auto k1_cnt = K1ft->cnt;
        const auto k2_raw_cnt = K2ft->cnt;
#ifdef SIMULATE
        // Reused camera streams can retain different counter origins.
        if (!camera_counter_offset_set) {
            k1_counter_origin = k1_cnt;
            k2_counter_origin = k2_raw_cnt;
            camera_counter_offset_set = true;
            info("Simulation camera counters aligned: K1=%lu K2=%lu",
                 k1_counter_origin, k2_counter_origin);
        }
        const auto k2_cnt = k2_raw_cnt - k2_counter_origin + k1_counter_origin;
#else
        const auto k2_cnt = k2_raw_cnt;
#endif
        // If we are here, then a new frame is available in both K1 and K2. 
        // Check that there has not been a counting error.
        if(k1_cnt == ft_cnt || k2_cnt == ft_cnt){
            reset_ddspc_fit("camera semaphore without a new frame", true);
            info("FT: Semaphore signalled but no new frame");
            nerrors++;
            continue;
        }
        const auto expected_cnt = ft_cnt + 1;
        ddspc_frame_gap = (k1_cnt != expected_cnt) ||
                          (k2_cnt != expected_cnt);
        // Check for missed frames
        if (k1_cnt > ft_cnt+2 || k2_cnt > ft_cnt+2){
            warn("Missed FT frames! K1: %lu K2: %lu FT: %lu",
                k1_cnt, k2_cnt, ft_cnt);
            // Catch up!
            while (sem_trywait(&K1ft->sem_new_frame)==0);
            while (sem_trywait(&K2ft->sem_new_frame)==0);
            if (k1_cnt > k2_cnt) ft_cnt = k2_cnt - 1;
            else ft_cnt = k1_cnt - 1;
            nerrors++;
        }
        ft_cnt++;
        int servo_mode;
        {
            std::lock_guard<std::mutex> lock(settings.mutex);
            servo_mode = settings.s.servo_mode;
        }
        if (servo_mode == SERVO_DDSPC && ddspc_frame_gap) {
            ++frame_gap_streak;
            ++frame_gaps_since_log;
            const auto now = std::chrono::steady_clock::now();
            if (now >= next_frame_gap_log) {
                int k1_pending = -1;
                int k2_pending = -1;
                if (sem_getvalue(&K1ft->sem_new_frame, &k1_pending) != 0)
                    k1_pending = -1;
                if (sem_getvalue(&K2ft->sem_new_frame, &k2_pending) != 0)
                    k2_pending = -1;
                warn("DDSPC camera frame gap: expected=%lu FFT K1=%lu K2=%lu "
                     "stream K1=%lu K2=%lu K2_raw=%lu FT_next=%lu "
                     "streak=%lu gaps_since_log=%lu pending K1=%d K2=%d "
                     "errors FT=%lu K1=%d K2=%d",
                     expected_cnt, k1_cnt, k2_cnt,
                     K1ft->subarray->md->cnt0, K2ft->subarray->md->cnt0,
                     k2_raw_cnt, ft_cnt,
                     frame_gap_streak, frame_gaps_since_log, k1_pending,
                     k2_pending, nerrors, K1ft->nerrors, K2ft->nerrors);
                next_frame_gap_log = now + std::chrono::seconds(1);
                frame_gaps_since_log = 0;
            }
        } else {
            if (servo_mode == SERVO_DDSPC && frame_gap_streak > 0) {
                const auto now = std::chrono::steady_clock::now();
                if (now >= next_frame_gap_log) {
                    info("DDSPC camera frame sequence recovered: K1=%lu K2=%lu "
                         "FT=%lu previous_gap_streak=%lu",
                         k1_cnt, k2_cnt, ft_cnt, frame_gap_streak);
                    next_frame_gap_log = now + std::chrono::seconds(1);
                }
            }
            frame_gap_streak = 0;
            if (servo_mode != SERVO_DDSPC) frame_gaps_since_log = 0;
        }
        bool ddspc_hold_this_frame =
            servo_mode == SERVO_DDSPC &&
            piston_reset_hold.on_paired_frame(!ddspc_frame_gap);
#ifdef PRINT_TIMING
        timespec then;
        clock_gettime(CLOCK_REALTIME, &then);
#endif
        // Extract the phases from the Fourier transforms, one baseline
        // at a time. This could in principle be vectorised. 
        gd_ix = ft_cnt % baselines.n_gd_boxcar;
        int pd_ix = ft_cnt % baselines.n_pd_boxcar;
        for (int bl=0; bl<N_BL; bl++){
            // Use the peak of the splodge to compute the phase
            x_px = lround(fs.x_px_K1[bl]) % K1ft->subim_sz;
            y_px = lround(fs.y_px_K1[bl]) % K1ft->subim_sz;
            stride = K1ft->subim_sz/2 + 1;
            K1_phasor[bl] = K1ft->ft[y_px*stride + x_px][0] + 
                1i*K1ft->ft[y_px*stride + x_px][1]*fs.sign[bl];
            // Also fill in the V^2 from the power spectrum.
            baselines.v2_K1(bl) = (K1ft->power_spectrum[y_px*stride + x_px]-K1ft->power_spectrum_bias)
                /K1ft->power_spectrum[0] * 16;

            // Fill in the boxcar average of the K1 phasor.
            baselines.pd_phasor_boxcar_avg(bl) -= baselines.pd_phasor_boxcar[pd_ix](bl);
            baselines.pd_phasor_boxcar[pd_ix](bl) = K1_phasor[bl];
            baselines.pd_phasor_boxcar_avg(bl) += baselines.pd_phasor_boxcar[pd_ix](bl);
            baselines.pd_av(bl) = std::arg(baselines.pd_phasor_boxcar_avg(bl)) /2/M_PI;
            
            x_px = lround(fs.x_px_K2[bl]) % K2ft->subim_sz;
            y_px = lround(fs.y_px_K2[bl]) % K2ft->subim_sz;
            stride = K2ft->subim_sz/2 + 1;
            K2_phasor[bl] = K2ft->ft[y_px*stride + x_px][0] + 
                1i*K2ft->ft[y_px*stride + x_px][1]*fs.sign[bl];
            // Also fill in the V^2 from the power spectrum.
            baselines.v2_K2(bl) = (K2ft->power_spectrum[y_px*stride + x_px]-K2ft->power_spectrum_bias)
                /K2ft->power_spectrum[0] * 16;

            // Compute the group delay - units of wavelengths at K1
            baselines.gd_phasor(bl) -= baselines.gd_phasor_boxcar[gd_ix](bl);
            baselines.gd_phasor_boxcar[gd_ix](bl) = 
                    K1_phasor[bl] * std::conj(K2_phasor[bl]);
            if (settings.s.offload_mode == OFFLOAD_MOD){
                // In mod mode, we skip the first frames of the boxcar.
                int dit_per_offload = 0.001 * settings.s.offload_time_ms / control_u.dit;
                if (control_u.nbreads > 1)
                	dit_per_offload += control_u.tsig_len;
                if (gd_ix < dit_per_offload){
                    baselines.gd_phasor_boxcar[gd_ix](bl) = 0;
                }
            } 
            baselines.gd_phasor(bl) += baselines.gd_phasor_boxcar[gd_ix](bl);  
            baselines.gd(bl) = std::arg(baselines.gd_phasor(bl)*baselines.gd_phasor_offset(bl)) * gd_to_K1;

            // Compute the unwrapped phase delay and signal to noise. 
            // NB We can only unwrap here if we are confident we have a algorithm that
            // can reverse this. It is difficult with 4 telescopes!
            // The phase delay is in units of the K1 central wavelength. 
            // For now... also have this feature with the Lacour algorithm.
            if ((servo_mode == SERVO_FIGHT) || (servo_mode == SERVO_SIMPLE)){
                // In fight mode, we just use the instantaneous phase, not the filtered phase.
                // This is useful for debugging, but not for real operation.
                // The 1.5 is a John Monnier hack, due to fmod's treatment of negative numbers.
                baselines.pd(bl) = std::fmod( (std::arg(K1_phasor[bl])/2/M_PI - baselines.pd_av_filtered(bl) + 1.5), 1.0) - 0.5;
            } else {
                // This is a raw (not unwrapped) phase.
                baselines.pd(bl) = std::arg(K1_phasor[bl])/2/M_PI; 
            }

            // Now we need the gd_snr and pd_snr for this baseline. 
            baselines.pd_snr(bl) = std::fabs(K1_phasor[bl])/std::sqrt(K1ft->power_spectrum_inst_bias);
            
            // Without boxcar averaging, the variance of the group delay phasor due to fundamental noise is:
            // Var(K1^* K2) = |K1|^2 Var(K2) + |K2|^2 Var(K1) + Var(K1) Var(K2)
            // where Var(K) = power_spectrum_bias. 
            // The GD_phasor has a variance sqrt(baselines[bl].n_gd_boxcar) larger than a
            // single phasor, so we need to divide by that. 
            baselines.gd_snr(bl) = std::fabs(baselines.gd_phasor(bl))/
                std::sqrt(K1ft->power_spectrum_bias * K2ft->power_spectrum_bias + 
                (K1ft->power_spectrum[y_px*stride + x_px] - K1ft->power_spectrum_bias)*K2ft->power_spectrum_bias +
                (K2ft->power_spectrum[y_px*stride + x_px] - K2ft->power_spectrum_bias)*K1ft->power_spectrum_bias)
                /std::sqrt(baselines.n_gd_boxcar);    
                
            // Set the weight matriix (bl,bl) to the square of the SNR, unless 
            // the SNR is too low, in which case we set it to zero.
            if ((baselines.gd_snr(bl) > settings.s.gd_threshold) && 
                (control_u.beams_active(baseline2beam[bl][0])) && (control_u.beams_active(baseline2beam[bl][1]))){
                Wgd(bl) = baselines.gd_snr(bl)*baselines.gd_snr(bl);
                var_gd(bl) = gd_to_K1*gd_to_K1/baselines.gd_snr(bl)/baselines.gd_snr(bl);
            }
            else {
                Wgd(bl) = 0;
                var_gd(bl) = 1e6; 
            }
            if ((baselines.pd_snr(bl) > settings.s.pd_threshold) &&
                (control_u.beams_active(baseline2beam[bl][0])) && (control_u.beams_active(baseline2beam[bl][1]))){
                Wpd(bl) = baselines.pd_snr(bl)*baselines.pd_snr(bl);
                var_pd(bl) = 1/baselines.pd_snr(bl)/baselines.pd_snr(bl)/4/M_PI/M_PI;
            }
            else{
                Wpd(bl) = 0;
                // If the SNR is too low, set the variance to something that is
                // practically infinite (i.e. 1000 wavelengths RMS here)
                var_pd(bl) = 1e6;
            }
        }
        
        // Now we have the group delays and phase delays, we can regularise by using by the  
        // I6gd matrix and the I6pd matrix. No short-cuts!
        // Fill a Vector of baseline group and phase delay.
        I6gd = M_lacour * make_pinv(Wgd, 0) * M_lacour.transpose() * Wgd.asDiagonal();
        I6pd = M_lacour * make_pinv(Wpd, 0) * M_lacour.transpose() * Wpd.asDiagonal();
        I4_search_projection = I4 - M_lacour_dag * I6gd * M_lacour; 
        gd_filtered = I6gd * baselines.gd;

        // Until SNR is high enough, pd_filtered is zero
#ifdef PRINT_TIMING_ALL
    clock_gettime(CLOCK_REALTIME, &then_all);
#endif
        if (servo_mode == SERVO_SIMPLE){
            pd_filtered += gd_filtered * settings.s.gd_gain;
        } else pd_filtered = filter6(I6pd, baselines.pd, Wpd);
#ifdef PRINT_TIMING_ALL
    clock_gettime(CLOCK_REALTIME, &now_all);
    if (then_all.tv_sec == now_all.tv_sec)
    info("PD filtering time: %ld", now_all.tv_nsec - then_all.tv_nsec);
#endif

        // Filter the average phase delay. !!! This doesn't work. Removing for now. !!!
        //baselines.pd_av_filtered = filter6(I6gd, baselines.pd_av);
        baselines.pd_av_filtered = baselines.pd_av;

        // The covariance matrix of baselines_gd and baselines_pd is given by a diagonal
        // matrix with the inverse of the SNR squared on the diagonal. We need to find the 
        // covariance of the telescope group and phase delays. 
        // !!! cov_pd_tel unused for now but could be useful?

        cov_gd_tel = M_lacour_dag * I6gd * var_gd.asDiagonal() * I6gd.transpose() * M_lacour_dag.transpose();

#ifdef DEBUG
        // Print debugging info for bugshooting, formatted for np.array input
        {
            std::ostringstream stream;
            stream << "var_gd diagonal = [";
            for (int k = 0; k < var_gd.size(); ++k) {
                stream << fmt::format("{:.6f}", var_gd(k)) << ((k < var_gd.size()-1) ? ", " : "]");
            }
            info("%s", stream.str().c_str());
        }
        {
            std::ostringstream stream;
            stream << "Wgd diagonal = [";
            for (int k = 0; k < Wgd.size(); ++k) {
                stream << fmt::format("{:.6f}", Wgd(k)) << ((k < Wgd.size()-1) ? ", " : "]");
            }
            info("%s", stream.str().c_str());
        }
        {
            std::ostringstream stream;
            stream << "cov_gd_tel diagonal = [";
            for (int k = 0; k < cov_gd_tel.diagonal().size(); ++k) {
                stream << fmt::format("{:.6f}", cov_gd_tel.diagonal()(k))
                       << ((k < cov_gd_tel.diagonal().size()-1) ? ", " : "]");
            }
            info("%s", stream.str().c_str());
        }
        {
            std::ostringstream stream;
            stream << "I6gd matrix = [\n";
            for (int i = 0; i < I6gd.rows(); ++i) {
                stream << "[";
                for (int j = 0; j < I6gd.cols(); ++j) {
                    stream << fmt::format("{:.6f}", I6gd(i, j)) << ((j < I6gd.cols()-1) ? ", " : "");
                }
                stream << "]" << ((i < I6gd.rows()-1) ? "," : "") << "\n";
            }
            stream << "]";
            info("%s", stream.str().c_str());
        }
#endif

        // Now project the filtered gd and pd onto telescope space.
        control_a.gd = M_lacour_dag * gd_filtered;
        control_a.pd = M_lacour_dag * pd_filtered;

        // Do the Fringe tracking! The error signal is the "delay" variable.
        // Only in this part do we ultiply by the K1 wavelength 
        // config["wave"]["K1"].value_or(2.05)

        // Based on whether there are fringe jumps, we may want to scale the pd gain.
        for (int i=0; i<N_TEL; i++){
            if ((cov_gd_tel(i,i) < GD_MAX_VAR_FOR_JUMP) && (cov_gd_tel(i,i) > GD_MIN_REAL_VAR)){
                if (std::fabs(control_a.gd(i)) > 0.5){
                    // We are more than 0.5 waves away, so we are likely to have a fringe jump.
                    // Set the pd gain scale to zero.
                    pd_gain_scale(i) = 0.0;
                } else {
                    // Scale the pd gain by the sinc of the gd offset, 
                    // so that if the gd is 0.5 waves away, the pd gain is zero.
                    pd_gain_scale(i) = sinc_normalized(control_a.gd(i));
                }
            }
        }

        const double wavelength_K1 = config["wave"]["K1"].value_or(2.05);
        const bool finite_servo_input = control_a.pd.allFinite() &&
                                        control_u.dm_piston.allFinite() &&
                                        std::isfinite(wavelength_K1) &&
                                        wavelength_K1 > 0.0;
        bool use_ddspc = false;
        double ddspc_regularization_before = 0.0;
        if (servo_mode == SERVO_DDSPC) {
            bool phase_connected = false;
            if (Wpd.allFinite()) {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> phase_solver(
                    M_lacour.transpose() * Wpd.asDiagonal() * M_lacour);
                phase_connected = phase_solver.eigenvalues().allFinite() &&
                                  phase_solver.eigenvalues()(1) > 1e-6;
            }
            const char *block_reason = nullptr;
            if (ddspc_hold_this_frame)
                block_reason = "piston settling after fit reset";
            else if (ddspc_frame_gap)
                block_reason = "camera frame gap";
            else if (!control_u.fringe_found) block_reason = "fringes not locked";
            else if (!(control_u.beams_active.minCoeff() > 0.5))
                block_reason = "inactive beam";
            else if (!(control_u.search.squaredNorm() == 0.0))
                block_reason = "fringe search active";
            else if (control_u.test_n != 0) block_reason = "DM test pattern active";
            else if (!phase_connected) block_reason = "phase baselines disconnected";
            else if (!finite_servo_input) block_reason = "non-finite servo input";
            // else if (last_gd_jump != 0 &&
            //          cnt_since_init <= last_gd_jump + 3)
            //     block_reason = "recent group-delay jump";
            use_ddspc = block_reason == nullptr;
            if (use_ddspc) {
                heimdallr_ddspc::Modes draw =
                    heimdallr_ddspc::Modes::Zero();
                if (!ddspc.frozen() &&
                    ddspc.exploration_frames() < ddspc_params.n_exploration) {
                    for (int i = 0; i < 3; ++i) {
                        draw(i) = standard_normal(exploration_generator);
                    }
                }
                ddspc_regularization_before = ddspc.controller().regularization();
                const Eigen::Vector4d command = ddspc.propose(
                    control_a.pd, control_u.dm_piston, wavelength_K1,
                    OPD_PER_DM_UNIT, MAX_DM_PISTON, draw);
                if (command.allFinite()) control_u.dm_piston = command;
                else {
                    use_ddspc = false;
                    block_reason = "non-finite proposed command";
                }
            }
            if (!use_ddspc) {
                reset_ddspc_fit(block_reason, false);
                if (!phase_connected) {
                    // Refresh the five-pair hold and zero this frame.
                    piston_reset_hold.arm();
                    piston_reset_hold.on_paired_frame(!ddspc_frame_gap);
                }
                ddspc_hold_this_frame =
                    ddspc_hold_this_frame || !phase_connected ||
                    piston_reset_hold.active();
            }
        }

        if (servo_mode == SERVO_SIMPLE){
           // Simple integrator, no fancy stuff. The group delay is added to the phase delay earlier.
            control_u.dm_piston += settings.s.kp * control_a.pd * config["wave"]["K1"].value_or(2.05)/OPD_PER_DM_UNIT;
            // Center the DM piston.
            control_u.dm_piston = control_u.dm_piston - control_u.dm_piston.mean()*Eigen::Vector4d::Ones();
            // Limit it to no more than +/- MAX_DM_PISTON.
            control_u.dm_piston = control_u.dm_piston.cwiseMin(MAX_DM_PISTON);
            control_u.dm_piston = control_u.dm_piston.cwiseMax(-MAX_DM_PISTON);
        }
        if (servo_mode==SERVO_FIGHT){
            // Compute the piezo control signal.
            control_u.dm_piston += (settings.s.kp * pd_gain_scale.asDiagonal() * control_a.pd +
                settings.s.gd_gain * control_a.gd) * config["wave"]["K1"].value_or(2.05)/OPD_PER_DM_UNIT;
            // Center the DM piston.
            control_u.dm_piston = control_u.dm_piston - control_u.dm_piston.mean()*Eigen::Vector4d::Ones();
            // Limit it to no more than +/- MAX_DM_PISTON.
            control_u.dm_piston = control_u.dm_piston.cwiseMin(MAX_DM_PISTON);
            control_u.dm_piston = control_u.dm_piston.cwiseMax(-MAX_DM_PISTON);

        } else if (servo_mode == SERVO_LACOUR ||
                   (servo_mode == SERVO_DDSPC && !use_ddspc &&
                    !ddspc_hold_this_frame)){
            // Compute the piezo control signal from the phase delay.
            if (servo_mode == SERVO_DDSPC && !finite_servo_input) {
                control_u.dm_piston.setZero();
            } else if ((cnt_since_init == last_gd_jump+1) ||
                       (cnt_since_init > last_gd_jump + 3)){
	        control_u.dm_piston += settings.s.kp * control_a.pd * config["wave"]["K1"].value_or(2.05)/OPD_PER_DM_UNIT;
            // Make sure that we only move the DM for for the active beams.
            control_u.dm_piston = control_u.beams_active.asDiagonal() * control_u.dm_piston;
           	// Center the DM piston.
            control_u.dm_piston = control_u.dm_piston - control_u.dm_piston.mean()*Eigen::Vector4d::Ones();
            // Limit it to no more than +/- MAX_DM_PISTON.
            control_u.dm_piston = control_u.dm_piston.cwiseMin(MAX_DM_PISTON);
            control_u.dm_piston = control_u.dm_piston.cwiseMax(-MAX_DM_PISTON);
            }
        }
        // Make the test pattern.
        if (control_u.test_n > 0 && !ddspc_hold_this_frame){
            if (control_u.test_ix < control_u.test_n){
                control_u.dm_piston(control_u.test_beam) = control_u.test_value;
            } else  {
                control_u.dm_piston(control_u.test_beam) = -control_u.test_value;
            } 
            control_u.test_ix = (control_u.test_ix + 1) % (2*control_u.test_n);
        }
        // Apply the signal to the DM! 
        control_u.dm_piston = piston_reset_hold.command(
            control_u.dm_piston, ddspc_hold_this_frame);
        set_dm_piston(control_u.dm_piston, ddspc_hold_this_frame);
        if (servo_mode == SERVO_DDSPC && use_ddspc) {
            if (control_u.test_n == 0 &&
                control_u.beams_active.minCoeff() > 0.5 &&
                control_u.search.squaredNorm() == 0.0) {
                // The active, search-free path wrote this clipped command unchanged.
                const bool was_frozen = ddspc.frozen();
                ddspc.update(control_u.dm_piston, wavelength_K1,
                             OPD_PER_DM_UNIT);
                if (!was_frozen && ddspc.frozen()) {
                    std::lock_guard<std::mutex> lock(settings.mutex);
                    settings.ddspc_freeze_status =
                        {false, true, ddspc.freeze_reason()};
                }
                const int updates = ddspc.controller().iterations();
                const int exploration = ddspc.exploration_frames();
                const double regularization = ddspc.controller().regularization();
                if (updates == 1) {
                    const auto now = std::chrono::steady_clock::now();
                    if (now >= next_ddspc_active_log) {
                        if (exploration <= ddspc_params.n_exploration &&
                            ddspc_params.n_exploration > 0) {
                            info("DDSPC active: exploration dither on (%d/%d), regularization %g -> %g",
                                 exploration, ddspc_params.n_exploration,
                                 ddspc_regularization_before,
                                 regularization);
                        } else {
                            info("DDSPC active: exploration complete, regularization %g -> %g",
                                 ddspc_regularization_before, regularization);
                        }
                        next_ddspc_active_log = now + std::chrono::seconds(1);
                    }
                } else if (regularization != ddspc_regularization_before) {
                    info("DDSPC regularization %g -> %g after %d valid updates; exploration %d/%d",
                         ddspc_regularization_before, regularization, updates,
                         exploration, ddspc_params.n_exploration);
                }
                if (ddspc_params.n_exploration > 0 &&
                    exploration == ddspc_params.n_exploration) {
                    info("DDSPC exploration complete after %d valid frames",
                         ddspc_params.n_exploration);
                }
            } else {
                reset_ddspc_fit("DM command path changed before update", false);
            }
        }

#ifdef PRINT_TIMING
        clock_gettime(CLOCK_REALTIME, &now);
        if (then.tv_sec == now.tv_sec)
            info("FT Computation time: %ld", now.tv_nsec - then.tv_nsec);
        then = now;
#endif
        // Phew! Now on to the less time-critical delay line control.

        // Apply the DL offload if enough time has passed since the last offload. !!! Remove 2 zeros.
        clock_gettime(CLOCK_REALTIME, &now);
        // Find time since last offload in milli-seconds as a double.
        double time_since_last_offload_ms = (now.tv_sec - last_dl_offload.tv_sec) * 1000.0 +
            (now.tv_nsec - last_dl_offload.tv_nsec) * 0.000001;
        // If it has been more than offload_time_ms, do the offload and the search step.
        // The exception is if we are in OFFLOAD_MOD mode, which we will treat separately for 
        // code readability.
        if ((settings.s.offload_mode == OFFLOAD_MOD) && (gd_ix == (int)baselines.n_gd_boxcar-1)){
            // In mod mode, we fill the group delay SNR matrix.
            gd_snr_during_mod.col(mod_ix) = baselines.gd_snr;
            mod_ix = (mod_ix + 1) % N_MOD;
            set_mod(MODULATION_AMPLITUDE * modulation_matrix.col(mod_ix));
            if (mod_ix==0){
                // Print out the full saved gd_snr.
                debug("GD SNR during mod:\n%s", log_stringify(gd_snr_during_mod).c_str());
                // Now find the fringe peak. We iterate over baselines, 
                // and accumulate the SNR for zero, plus and minus modulation.
                Eigen::Matrix<double, N_BL, 1> delays;
                delays.setZero();
                Eigen::Matrix<double, N_BL, 1> valid;
                valid.setZero();
                for (int bl=0; bl<N_BL; bl++){
                    double snr_zero = 0;
                    double snr_plus = 0;
                    double snr_minus = 0;
                    for (int ix=0; ix<N_MOD; ix++){
                        if (bl_modulation_matrix(bl,ix) == 0) snr_zero += gd_snr_during_mod(bl,ix);
                        else if (bl_modulation_matrix(bl,ix) == 1) snr_plus += gd_snr_during_mod(bl,ix);
                        else if (bl_modulation_matrix(bl,ix) == -1) snr_minus += gd_snr_during_mod(bl,ix);
                    }
                    snr_zero /= bzn[bl]; //!!! 6 for N_MOD=14
                    snr_plus /= bmodn[bl]; //!!! 4 for N_MOD=14
                    snr_minus /= bmodn[bl]; //!!! 4 for N_MOD=14
                    // DEBUG
                    debug("%.1f %.1f %.1f", snr_minus, snr_zero, snr_plus);
                    // Find max SNR and corresponding delay.
                    double max_snr = std::max({snr_zero, snr_plus, snr_minus});
                    if (max_snr > settings.s.gd_threshold){
                        valid(bl) = 1;
                        if (max_snr == snr_plus) delays(bl) = -MODULATION_AMPLITUDE;
                        else if (max_snr == snr_minus) delays(bl) = MODULATION_AMPLITUDE;
                    }
                }
                // Create a new pseudo-inverse matrix, and multiply the group delay 
                // by this to find the new control signal. There is regularisation just like
                // the normal fringe tracking above.
                debug("Delays: %s", log_stringify(delays.transpose()).c_str());
                debug("Valid: %s", log_stringify(valid.transpose()).c_str());
                I6gd = M_lacour * make_pinv(valid, 0) * M_lacour.transpose() * valid.asDiagonal();
                control_u.dl_offload = M_lacour_dag * I6gd * delays;
                debug("Regularised: %s", log_stringify((I6gd * delays).transpose()).c_str());
                debug("Telescope Space: %s", log_stringify(control_u.dl_offload.transpose()).c_str());
                add_to_delay_lines(control_u.search - control_u.dl_offload);
            }
        } else if (time_since_last_offload_ms > settings.s.offload_time_ms) {
            // Irrespective of offload type, see if we need to reset the search, 
            // based on determining if we confidently have fringes with all telescopes.
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N_TEL, N_TEL>> eig_solver(cov_gd_tel);
            double worst_gd_var = eig_solver.eigenvalues().maxCoeff();
            double gd_var_threshold = gd_to_K1*gd_to_K1/settings.s.gd_search_reset/settings.s.gd_search_reset;

            // Find the nth smallest eigenvalue, where n=2 if all
            // telescopes are active, and n increases by 1 for each inactive telescope.
            beam_mutex.lock();
            unsigned int num_zeros = 0;
            for (int i = 0; i < N_TEL; ++i) {
                if (control_u.beams_active(i) == 0) num_zeros++;
            }
            unsigned int n = 2 + num_zeros;

            // Find the nth minimum eigenvalue (n is 1-based, so n=1 is the smallest)
            Eigen::VectorXd evals = eig_solver.eigenvalues();
            std::vector<double> eval_vec(evals.data(), evals.data() + evals.size());
            std::sort(eval_vec.begin(), eval_vec.end());
            double nth_min_eval = eval_vec.size() >= n ? eval_vec[n-1] : eval_vec.back();


            if ((worst_gd_var < gd_var_threshold) && (nth_min_eval > GD_MIN_REAL_VAR)){
                control_u.search_Nsteps=0;
                control_u.search.setZero();
                control_u.fringe_found = true;
                control_u.itime += settings.s.offload_time_ms/1000.0;
                //fmt::print("Resetting search, good fringes detected. GD vars: {:.4f} {:.4f} {:.4f} {:.4f}\n", 
                //	cov_gd_tel.diagonal()(0), cov_gd_tel.diagonal()(1),cov_gd_tel.diagonal()(2), cov_gd_tel.diagonal()(3));
                 //           fmt::print("cov_gd_tel eigenvalues: ");
                //for (int i = 0; i < eig_solver.eigenvalues().size(); ++i) {
                // fmt::print("{:.6f}{}", eig_solver.eigenvalues()(i), (i < eig_solver.eigenvalues().size()-1) ? ", " : "\n");
                //}
                //fmt::print("GD var threshold: {:.6f}\n", gd_var_threshold);
            } else {
                if (foreground_in_place) control_u.itime += settings.s.offload_time_ms/1000.0;
                control_u.fringe_found = false;
                // Now do the delay line control. This is slower, so occurs after the servo.
                // Compute the search sign.
                unsigned int search_level = 0;
                unsigned int index = control_u.search_Nsteps/control_u.steps_to_turnaround + 1;
                //This gives a logarithm base 2, so we search twice as far each turnaround. 
                while (index >>= 1) ++search_level;
                control_u.search = I4_search_projection *control_u.search_delta * (1.0 - (search_level % 2) * 2.0)
                    * control_u.beams_active.asDiagonal() * search_vector_scale;
               control_u.search_Nsteps++;
            }

            // if testauto now_n is negative, over-write this with a test pattern.
            if (control_u.test_n < 0){
                control_u.search.setZero();
                // Set the test_beam to have a square wave of amplitude test_value, half period 1 offloads.
                if (control_u.test_ix % 2 == 0){
                    control_u.search(control_u.test_beam) = control_u.test_value;
                } else {
                    control_u.search(control_u.test_beam) = -control_u.test_value;
                }
                control_u.test_ix = (control_u.test_ix + 1) % 2;
            }

            if ((settings.s.offload_mode == OFFLOAD_NESTED) && (servo_mode != SERVO_OFF)){
            	// Add to the delay line offload.
        	    control_u.dl_offload = 0.3*control_u.dm_piston * OPD_PER_DM_UNIT;
                if (((servo_mode == SERVO_LACOUR) ||
                     (servo_mode == SERVO_DDSPC)) &&
                    (cnt_since_init - last_gd_jump > baselines.n_gd_boxcar)){
                    // Use the group delay to make full fringe jumps, only if there has been at least
                    // baselines.n_gd_boxcar frames since initialisation or the last jump.
            	    for (int i=0; i<N_TEL; i++){
                        if (cov_gd_tel(i,i) < GD_MAX_VAR_FOR_JUMP) {
                            if (std::fabs(control_a.gd(i)) > 0.75){ 
                                control_u.dl_offload(i) += sgn(control_a.gd(i))*config["wave"]["K1"].value_or(2.05);
                                last_gd_jump = cnt_since_init;
                            } 
                        }
                    }
                }
                //fmt::print("Offload: {} {} {} {}\n", control_u.dl_offload(0), control_u.dl_offload(1),
                //   control_u.dl_offload(2), control_u.dl_offload(3));
                //fmt::print("Search: {} {} {} {}\n", control_u.search(0), control_u.search(1),
                //   control_u.search(2), control_u.search(3));
                
                add_to_delay_lines(control_u.search - control_u.dl_offload);
                //control_u.dl_offload.setZero();
            }
            else if (settings.s.offload_mode == OFFLOAD_GD) {
                add_to_delay_lines(control_u.search - settings.s.offload_gd_gain*control_a.gd * config["wave"]["K1"].value_or(2.05));
            }
            beam_mutex.unlock();
            last_dl_offload = now;
            sem_post(&sem_offload);
        }
   
        // Now we sanity check by computing the bispectrum and closure phases.
        // K1 and K2, in case of tracking on resolved objects... 
        for (int cp=0; cp<N_CP; cp++){
            int K1_ix = bispectra_K1[cp].ix_bs_boxcar;
            int K2_ix = bispectra_K2[cp].ix_bs_boxcar;
            int bl1 = closure2bl[cp][0];
            int bl2 = closure2bl[cp][1];
            int bl3 = closure2bl[cp][2];
            bispectra_K1[cp].bs_phasor -= bispectra_K1[cp].bs_phasors[K1_ix];
            bispectra_K2[cp].bs_phasor -= bispectra_K2[cp].bs_phasors[K2_ix];
            bispectra_K1[cp].bs_phasors[K1_ix] = 
                K1_phasor[bl1] * K1_phasor[bl2] * std::conj(K1_phasor[bl3]);
            bispectra_K2[cp].bs_phasors[K2_ix] =
                K2_phasor[bl1] * K2_phasor[bl2] * std::conj(K2_phasor[bl3]);
            bispectra_K1[cp].bs_phasor += bispectra_K1[cp].bs_phasors[K1_ix];
            bispectra_K2[cp].bs_phasor += bispectra_K2[cp].bs_phasors[K2_ix];
            // Compute the closure phase.
            bispectra_K1[cp].closure_phase = std::arg(bispectra_K1[cp].bs_phasor);
            bispectra_K2[cp].closure_phase = std::arg(bispectra_K2[cp].bs_phasor);
            // Increment the counters.
            bispectra_K1[cp].ix_bs_boxcar = (bispectra_K1[cp].ix_bs_boxcar + 1) % bispectra_K1[cp].n_bs_boxcar;
            bispectra_K2[cp].ix_bs_boxcar = (bispectra_K2[cp].ix_bs_boxcar + 1) % bispectra_K2[cp].n_bs_boxcar;
            //std::cout << "CP: " << cp << " Phase: " << bispectra[cp].closure_phase << std::endl;
        }
    }
}
