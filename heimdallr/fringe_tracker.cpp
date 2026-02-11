#include "heimdallr.h"
//#define PRINT_TIMING
//#define PRINT_TIMING_ALL
//#define DEBUG
//#define DEBUG_FILTER6

// Thresholds for fringe tracking (now variables)
double gd_threshold = 8.0;
double pd_threshold = 4.5;
double gd_search_reset = 6.0;

#define MAX_DM_PISTON 0.4
// Group delay is in wavelengths at 2.05 microns. Need 0.5 waves to be 2.5 sigma.
#define GD_MAX_VAR_FOR_JUMP 0.2*0.2
#define GD_MIN_REAL_VAR 1E-6

using namespace std::complex_literals;

long unsigned int ft_cnt=0, cnt_since_init=0;
long unsigned int nerrors=0;
double gd_to_K1=1.0;

// Local baseline variables
double x_px_K1[N_BL], y_px_K1[N_BL], x_px_K2[N_BL], y_px_K2[N_BL], sign[N_BL];
dcomp K1_phasor[N_BL], K2_phasor[N_BL];

// A 6x6 matrix for the weights of phase and group delay
Eigen::Matrix<double, N_BL, 1> var_pd, var_gd, Wpd, Wgd;
Eigen::Matrix<double, N_BL, 1> pd_filtered, gd_filtered;

// Convenience matrices and vectors.
// A 4x4 matrix of zeros to store the diagonal.
Eigen::Matrix<double, N_TEL, N_TEL> singularDiag = Eigen::Matrix<double, N_TEL, N_TEL>::Zero();

// A 4x4 identity matrix.
Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();

// The search vector.
Eigen::Vector4d search_vector_scale(-2.75,-1.75,1.25,3.25);

// Beams active as an Eigen vector (i.e. doubles rather than booleans)
Eigen::Vector4d beams_active = Eigen::Vector4d::Ones();

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
        std::cout << "SVD time: " << now.tv_nsec-then.tv_nsec << std::endl;
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
        std::cout << "Thresholding time: " << now.tv_nsec-then.tv_nsec << std::endl;
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

void set_dm_piston(Eigen::Vector4d dm_piston){
#ifdef SIMULATE
    return;
#endif
    // Make sure that we only move the DM for for the active beams.
    control_u.dm_piston = beams_active.asDiagonal() * control_u.dm_piston;       	
    // This function sets the DM piston to the given value.
    for(int i = 0; i < N_TEL; i++) {
        if (control_u.search(i) != 0.0) {
            control_u.dm_piston(i) = 0.0; // Reset DM piston if in search mode
            continue; // Do not set if in search mode
        }
        for (int j=0; j<144; j++){
            DMs[i].array.D[j] = dm_piston(i);
        }
        ImageStreamIO_sempost(&master_DMs[i], 1);
    }
}

// Initialise variables assocated with baselines, including 
// bispectra.
void initialise_baselines(){
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
    baselines.set_gd_boxcar(64);
    baselines.n_pd_boxcar=MAX_N_PD_BOXCAR;
    baselines.pd_phasor.setZero();
    baselines.pd_phasor_boxcar_avg.setZero();
    baselines.pd_av_filtered.setZero();
    baselines.pd_av.setZero();

    // Reset the boxcar averages
    //for (int i=0; i<baselines.n_gd_boxcar; i++){
    //    baselines.gd_phasor_boxcar[i].setZero();
    //}

    for (int i=0; i<baselines.n_pd_boxcar; i++){
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
    float pix = config["geometry"]["pix"].value_or(24.0);
    float wave_K1 = config["wave"]["K1"].value_or(2.05);
    float wave_K2 = config["wave"]["K2"].value_or(2.25);
    gd_to_K1 = wave_K2/(wave_K2-wave_K1)/2/M_PI;

    for (int bl=0; bl<N_BL; bl++){
        // Set the offsets to the group delay
        baselines.gd_phasor_offset(bl) = 
            std::exp(-1.0i *config["servo"]["gd_phasor_offset"][bl].value_or(0.0)/gd_to_K1);

        // Set the x and y coordinates for extracting flux
        float bl_x = config["geometry"]["beam_x"][baseline2beam[bl][1]].value_or(0.0) -
            config["geometry"]["beam_x"][baseline2beam[bl][0]].value_or(0.0);
        float bl_y = config["geometry"]["beam_y"][baseline2beam[bl][1]].value_or(0.0) -
            config["geometry"]["beam_y"][baseline2beam[bl][0]].value_or(0.0);
        if (bl_x < 0){
            bl_x = -bl_x;
            bl_y = -bl_y;
            sign[bl] = -1;
        } else sign[bl] = 1;
        x_px_K1[bl] = bl_x * pix / wave_K1 * K1ft->subim_sz;
        y_px_K1[bl] = bl_y * pix / wave_K1 * K1ft->subim_sz;
        x_px_K2[bl] = bl_x * pix / wave_K2 * K2ft->subim_sz;
        y_px_K2[bl] = bl_y * pix / wave_K2 * K2ft->subim_sz;
        if (bl_y < 0){
            y_px_K1[bl] += K1ft->subim_sz;
            y_px_K2[bl] += K2ft->subim_sz;
        }
        //std::cout << "Baseline: " << bl << " x_px_K1: " << x_px_K1[bl] << " y_px_K1: " << y_px_K1[bl] << std::endl;
        //std::cout << "Baseline: " << bl << " x_px_K2: " << x_px_K2[bl] << " y_px_K2: " << y_px_K2[bl] << std::endl;
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
    control_u.target_itime=0;
    beam_mutex.unlock();

    pid_settings.mutex.lock();
    pid_settings.kp = 0.5;
    pid_settings.ki = 0.0;
    pid_settings.kd = 0.0;
    pid_settings.integral = 0.0;
    pid_settings.dl_feedback_gain = 0.0;
    pid_settings.gd_gain = pid_settings.kp / baselines.n_gd_boxcar;
    pid_settings.offload_gd_gain = 1.0;
    pid_settings.mutex.unlock();
}

// This takes 12 microseconds on average. A little slow!
Eigen::Matrix<double, N_BL, 1> filter6(Eigen::Matrix<double, N_BL, N_BL> I6, Eigen::Matrix<double, N_BL, 1> x, Eigen::Matrix<double, N_BL, 1> W){
    // This function filters the input vector x using the I6gd matrix.
    // It returns the filtered vector.
    double chi2=1e6, chi2_min=1e6;
    int i_best;
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
            i_best = i;
        }
    }
#ifdef DEBUG_FILTER6
    // For debugging, print the best combination found, x_best, and y_best
    fmt::print("Best i {:b}\n", i_best);
    // Print Eigen vectors as comma-separated values
    fmt::print("Initial x:       ");
    for (int k = 0; k < x.size(); ++k) {
        fmt::print("{:.4f}{}", x(k), (k < x.size()-1) ? ", " : "\n");
    }
    fmt::print("Best modified x: ");
    for (int k = 0; k < x_best.size(); ++k) {
        fmt::print("{:.4f}{}", x_best(k), (k < x_best.size()-1) ? ", " : "\n");
    }
    fmt::print("Best y:           ");
    for (int k = 0; k < y_best.size(); ++k) {
        fmt::print("{:.4f}{}", y_best(k), (k < y_best.size()-1) ? ", " : "\n");
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

    long x_px, y_px, stride;
    initialise_baselines();
    reset_search();
    set_dm_piston(Eigen::Vector4d::Zero()); 
    ft_cnt = K1ft->cnt;
    while(servo_mode != SERVO_STOP){
        cnt_since_init++; //This should "never" wrap around, as a long int is big.
        // See if there was a semaphore signalled for the next frame to be ready in K1 and K2
        sem_wait(&K1ft->sem_new_frame);
        sem_wait(&K2ft->sem_new_frame);
        if ((K1ft->bad_frame) || (K1ft->bad_frame)) {
            ft_cnt++;
            continue;
        }
        // If we are here, then a new frame is available in both K1 and K2. 
        // Check that there has not been a counting error.
        if(K1ft->cnt == ft_cnt || K2ft->cnt == ft_cnt){
            std::cout << "FT: Semaphore signalled but no new frame" << std::endl;
            nerrors++;
            continue;
        }
        // Check for missed frames
        if (K1ft->cnt > ft_cnt+2 || K2ft->cnt > ft_cnt+2){
            std::cout << "Missed FT frames! K1: " << K1ft->cnt << " K2: " 
                << K2ft->cnt << " FT: " << ft_cnt << std::endl;
            // Catch up!
            while (sem_trywait(&K1ft->sem_new_frame)==0);
            while (sem_trywait(&K2ft->sem_new_frame)==0);
            if (K1ft->cnt > K2ft->cnt) ft_cnt = K2ft->cnt - 1;
            else ft_cnt = K1ft->cnt - 1;
            nerrors++;
        }
        ft_cnt++;
#ifdef PRINT_TIMING
        timespec then;
        clock_gettime(CLOCK_REALTIME, &then);
#endif
        // Extract the phases from the Fourier transforms, one baseline
        // at a time. This could in principle be vectorised. 
        int gd_ix = cnt % baselines.n_gd_boxcar;
        int pd_ix = cnt % baselines.n_pd_boxcar;
        for (int bl=0; bl<N_BL; bl++){
            // Use the peak of the splodge to compute the phase
            x_px = lround(x_px_K1[bl]) % K1ft->subim_sz;
            y_px = lround(y_px_K1[bl]) % K1ft->subim_sz;
            stride = K1ft->subim_sz/2 + 1;
            K1_phasor[bl] = K1ft->ft[y_px*stride + x_px][0] + 
                1i*K1ft->ft[y_px*stride + x_px][1]*sign[bl];
            // Also fill in the V^2 from the power spectrum.
            baselines.v2_K1(bl) = (K1ft->power_spectrum[y_px*stride + x_px]-K1ft->power_spectrum_bias)
                /K1ft->power_spectrum[0] * 16;

            // Fill in the boxcar average of the K1 phasor.
            baselines.pd_phasor_boxcar_avg(bl) -= baselines.pd_phasor_boxcar[pd_ix](bl);
            baselines.pd_phasor_boxcar[pd_ix](bl) = K1_phasor[bl];
            baselines.pd_phasor_boxcar_avg(bl) += baselines.pd_phasor_boxcar[pd_ix](bl);
            baselines.pd_av(bl) = std::arg(baselines.pd_phasor_boxcar_avg(bl)) /2/M_PI;
            
            x_px = lround(x_px_K2[bl]) % K2ft->subim_sz;
            y_px = lround(y_px_K2[bl]) % K2ft->subim_sz;
            stride = K2ft->subim_sz/2 + 1;
            K2_phasor[bl] = K2ft->ft[y_px*stride + x_px][0] + 
                1i*K2ft->ft[y_px*stride + x_px][1]*sign[bl];
            // Also fill in the V^2 from the power spectrum.
            baselines.v2_K2(bl) = (K2ft->power_spectrum[y_px*stride + x_px]-K2ft->power_spectrum_bias)
                /K2ft->power_spectrum[0] * 16;

            // Compute the group delay - units of wavelengths at K1
            baselines.gd_phasor(bl) -= baselines.gd_phasor_boxcar[gd_ix](bl);
            baselines.gd_phasor_boxcar[gd_ix](bl) = 
                K1_phasor[bl] * std::conj(K2_phasor[bl]);
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
                (K2ft->power_spectrum[y_px*stride + x_px] - K2ft->power_spectrum_bias)*K1ft->power_spectrum_bias)/
                std::sqrt(baselines.n_gd_boxcar);    
                
            // Set the weight matriix (bl,bl) to the square of the SNR, unless 
            // the SNR is too low, in which case we set it to zero.
            if ((baselines.gd_snr(bl) > gd_threshold) && 
                (control_u.beams_active[baseline2beam[bl][0]]) && (control_u.beams_active[baseline2beam[bl][1]])){
                Wgd(bl) = baselines.gd_snr(bl)*baselines.gd_snr(bl);
                var_gd(bl) = gd_to_K1*gd_to_K1/baselines.gd_snr(bl)/baselines.gd_snr(bl);
            }
            else {
                Wgd(bl) = 0;
                var_gd(bl) = 1e6; 
            }
            if ((baselines.pd_snr(bl) > pd_threshold) &&
                (control_u.beams_active[baseline2beam[bl][0]]) && (control_u.beams_active[baseline2beam[bl][1]])){
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
        
        // Make the beams_active a vector.
        beams_active = Eigen::Vector4d(control_u.beams_active[0],control_u.beams_active[1],control_u.beams_active[2],control_u.beams_active[3]);

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
            pd_filtered += gd_filtered * pid_settings.gd_gain;
        } else pd_filtered = filter6(I6pd, baselines.pd, Wpd);
#ifdef PRINT_TIMING_ALL
    clock_gettime(CLOCK_REALTIME, &now_all);
    if (then_all.tv_sec == now_all.tv_sec)
        std::cout << "PD filtering time: " << now_all.tv_nsec-then_all.tv_nsec << std::endl;
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
        fmt::print("var_gd diagonal = [");
        for (int k = 0; k < var_gd.size(); ++k) {
            fmt::print("{:.6f}{}", var_gd(k), (k < var_gd.size()-1) ? ", " : "]\n");
        }
        fmt::print("Wgd diagonal = [");
        for (int k = 0; k < Wgd.size(); ++k) {
            fmt::print("{:.6f}{}", Wgd(k), (k < Wgd.size()-1) ? ", " : "]\n");
        }
        fmt::print("cov_gd_tel diagonal = [");
        for (int k = 0; k < cov_gd_tel.diagonal().size(); ++k) {
            fmt::print("{:.6f}{}", cov_gd_tel.diagonal()(k), (k < cov_gd_tel.diagonal().size()-1) ? ", " : "]\n");
        }
        fmt::print("I6gd matrix = [\n");
        for (int i = 0; i < I6gd.rows(); ++i) {
            fmt::print("[");
            for (int j = 0; j < I6gd.cols(); ++j) {
                fmt::print("{:.6f}{}", I6gd(i, j), (j < I6gd.cols()-1) ? ", " : "");
            }
            fmt::print("]{}\n", (i < I6gd.rows()-1) ? "," : "");
        }
        fmt::print("]\n");
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

        if (servo_mode == SERVO_SIMPLE){
           // Simple integrator, no fancy stuff. The group delay is added to the phase delay earlier.
            control_u.dm_piston += pid_settings.kp * control_a.pd * config["wave"]["K1"].value_or(2.05)/OPD_PER_DM_UNIT;
            // Center the DM piston.
            control_u.dm_piston = control_u.dm_piston - control_u.dm_piston.mean()*Eigen::Vector4d::Ones();
            // Limit it to no more than +/- MAX_DM_PISTON.
            control_u.dm_piston = control_u.dm_piston.cwiseMin(MAX_DM_PISTON);
            control_u.dm_piston = control_u.dm_piston.cwiseMax(-MAX_DM_PISTON);
        }
        if (servo_mode==SERVO_FIGHT){
            // Compute the piezo control signal.
            control_u.dm_piston += (pid_settings.kp * pd_gain_scale.asDiagonal() * control_a.pd +
                pid_settings.gd_gain * control_a.gd) * config["wave"]["K1"].value_or(2.05)/OPD_PER_DM_UNIT;
            // Center the DM piston.
            control_u.dm_piston = control_u.dm_piston - control_u.dm_piston.mean()*Eigen::Vector4d::Ones();
            // Limit it to no more than +/- MAX_DM_PISTON.
            control_u.dm_piston = control_u.dm_piston.cwiseMin(MAX_DM_PISTON);
            control_u.dm_piston = control_u.dm_piston.cwiseMax(-MAX_DM_PISTON);

        } else if (servo_mode == SERVO_LACOUR){
            // Compute the piezo control signal from the phase delay.
            if ((cnt_since_init == last_gd_jump+1) || (cnt_since_init > last_gd_jump + 3)){
	        control_u.dm_piston += pid_settings.kp * control_a.pd * config["wave"]["K1"].value_or(2.05)/OPD_PER_DM_UNIT;
            // Make sure that we only move the DM for for the active beams.
            control_u.dm_piston = beams_active.asDiagonal() * control_u.dm_piston;
           	// Center the DM piston.
            control_u.dm_piston = control_u.dm_piston - control_u.dm_piston.mean()*Eigen::Vector4d::Ones();
            // Limit it to no more than +/- MAX_DM_PISTON.
            control_u.dm_piston = control_u.dm_piston.cwiseMin(MAX_DM_PISTON);
            control_u.dm_piston = control_u.dm_piston.cwiseMax(-MAX_DM_PISTON);
            }
        }
        // Make the test pattern.
        if (control_u.test_n > 0){
            if (control_u.test_ix < control_u.test_n){
                control_u.dm_piston(control_u.test_beam) = control_u.test_value;
            } else  {
                control_u.dm_piston(control_u.test_beam) = -control_u.test_value;
            } 
            control_u.test_ix = (control_u.test_ix + 1) % (2*control_u.test_n);
        }
        // Apply the signal to the DM! 
        set_dm_piston(control_u.dm_piston);

#ifdef PRINT_TIMING
        clock_gettime(CLOCK_REALTIME, &now);
        if (then.tv_sec == now.tv_sec)
            std::cout << "FT Computation time: " << now.tv_nsec-then.tv_nsec << std::endl;
        then = now;
#endif
        // Phew! Now on to the less time-critical delay line control.

        // Apply the DL offload if enough time has passed since the last offload. !!! Remove 2 zeros.
        clock_gettime(CLOCK_REALTIME, &now);
        // Find time since last offload in milli-seconds as a double.
        double time_since_last_offload_ms = (now.tv_sec - last_dl_offload.tv_sec) * 1000.0 +
            (now.tv_nsec - last_dl_offload.tv_nsec) * 0.000001;
        // If it has been more than offload_time_ms, do the offload and the search step.
        if (time_since_last_offload_ms > offload_time_ms){
            // Irrespective of offload type, see if we need to reset the search, 
            // based on determining if we confidently have fringes with all telescopes.
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N_TEL, N_TEL>> eig_solver(cov_gd_tel);
            double worst_gd_var = eig_solver.eigenvalues().maxCoeff();
            double gd_var_threshold = gd_to_K1*gd_to_K1/gd_search_reset/gd_search_reset;

            // Find the nth smallest eigenvalue, where n=2 if all
            // telescopes are active, and n increases by 1 for each inactive telescope.

            unsigned int num_zeros = 0;
            for (int i = 0; i < N_TEL; ++i) {
                if (control_u.beams_active[i] == 0) num_zeros++;
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
                control_u.itime += offload_time_ms/1000.0;
                //fmt::print("Resetting search, good fringes detected. GD vars: {:.4f} {:.4f} {:.4f} {:.4f}\n", 
                //	cov_gd_tel.diagonal()(0), cov_gd_tel.diagonal()(1),cov_gd_tel.diagonal()(2), cov_gd_tel.diagonal()(3));
                 //           fmt::print("cov_gd_tel eigenvalues: ");
                //for (int i = 0; i < eig_solver.eigenvalues().size(); ++i) {
                // fmt::print("{:.6f}{}", eig_solver.eigenvalues()(i), (i < eig_solver.eigenvalues().size()-1) ? ", " : "\n");
                //}
                //fmt::print("GD var threshold: {:.6f}\n", gd_var_threshold);
            } else {
                if (foreground_in_place) control_u.itime += offload_time_ms/1000.0;
                control_u.fringe_found = false;
                // Now do the delay line control. This is slower, so occurs after the servo.
                // Compute the search sign.
                unsigned int search_level = 0;
                unsigned int index = control_u.search_Nsteps/control_u.steps_to_turnaround + 1;
                //This gives a logarithm base 2, so we search twice as far each turnaround. 
                while (index >>= 1) ++search_level;
                control_u.search = I4_search_projection *control_u.search_delta * (1.0 - (search_level % 2) * 2.0)
                    * beams_active.asDiagonal() * search_vector_scale;
                control_u.search -= (beams_active.asDiagonal() * control_u.search).mean() * beams_active;
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

            if ((offload_mode == OFFLOAD_NESTED) && (servo_mode!=SERVO_OFF)){
            	// Add to the delay line offload.
        	    control_u.dl_offload = 0.3*control_u.dm_piston * OPD_PER_DM_UNIT;
            	if ((servo_mode == SERVO_LACOUR) && (cnt_since_init - last_gd_jump > baselines.n_gd_boxcar)){
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
            else if (offload_mode == OFFLOAD_GD) {
                add_to_delay_lines(control_u.search - pid_settings.offload_gd_gain*control_a.gd * config["wave"]["K1"].value_or(2.05));
            }
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
