#!/usr/bin/env python3
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_telemetry(json_file, top_fraction):
    with open(json_file, 'r') as f:
        data = json.load(f)
    ts = np.array(data['timestamps_ms'])  # timestamps in ms
    poke = np.array(data['poke_values']) > 0
    imgs = np.array(data['images'])
    mean_off = imgs[~poke].mean(axis=0)
    mean_on = imgs[poke].mean(axis=0)
    diff = np.abs(mean_on - mean_off)
    sel = np.argsort(diff)[-int(top_fraction * imgs.shape[1]):]
    sig = imgs[:, sel].mean(axis=1)
    return ts, poke, sig

def average_impulse(sig, poke, window):
    onsets = np.where(np.diff(np.concatenate(([0], poke.astype(int)))) == 1)[0]
    windows = []
    for o in onsets:
        if o + window <= len(sig):
            windows.append(sig[o:o+window])
    return np.array(windows).mean(axis=0)

def fit_fopdt(h, Ts):
    s = np.cumsum(h) * Ts
    t = np.arange(len(s)) * Ts
    def fopdt(t, L, T, K):
        y = K * (1 - np.exp(-(t - L) / T))
        y[t < L] = 0
        return y
    p0 = [Ts, 5 * Ts, s[-1]]
    popt, _ = curve_fit(fopdt, t, s, p0=p0, bounds=([0, 0, 0], [10 * Ts, 10 * Ts, s[-1] * 10]))
    return popt  # L, T, Kss

def design_pid_zn(L, T):
    Kp = 1.2 * (T / L)
    Ti = 2 * L
    Td = 0.5 * L
    return Kp, Ti, Td

def estimate_bandwidth(h, Ts0, oversample):
    nfft = 512
    Hf = np.fft.rfft(h, nfft)
    freqs = np.fft.rfftfreq(nfft, Ts0)
    mag = np.abs(Hf)
    H0 = mag[0]
    thr = H0 / np.sqrt(2)
    idx = np.where((mag <= thr) & (freqs > 0))[0]
    if len(idx) > 0:
        fc = freqs[idx[0]]
        fs_opt = oversample * fc
        return fc, fs_opt, freqs, mag
    else:
        return None, None, freqs, mag
def main():
    parser = argparse.ArgumentParser(description="Estimate TF and tune PID with optimal fs")
    parser.add_argument("json_file", help="Path to latency_telem.json")
    parser.add_argument("--top_fraction", type=float, default=0.05)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--oversample", type=float, default=15.0,
                        help="Factor above fc for sampling rate")
    args = parser.parse_args()

    ts, poke, sig = load_telemetry(args.json_file, args.top_fraction)
    dt_ms = np.median(np.diff(ts))  # median spacing in milliseconds
    Ts0 = dt_ms / 1000.0            # convert to seconds

    h = average_impulse(sig, poke, args.window)

    # Correct impulse plot: time vector in milliseconds
    tvec_ms = np.arange(len(h)) * Ts0 * 1000  # x-axis in ms
    plt.figure()
    plt.plot(tvec_ms, h, '-o')
    plt.xlabel('Time [ms]')
    plt.ylabel('Amplitude [a.u.]')
    plt.title('Impulse Response')
    plt.grid(True)
    plt.savefig('/home/asg/Music/impulse.png')
    plt.close()

    # Bandwidth and optimal fs
    fc, fs_opt, freqs, mag = estimate_bandwidth(h, Ts0, args.oversample)
    if fc:
        Ts = 1 / fs_opt
        print(f"-3dB bandwidth: {fc:.2f} Hz; suggested fs={fs_opt:.1f} Hz => Ts={Ts*1000:.2f} ms")
    else:
        Ts = Ts0
        print("Could not determine bandwidth; using original Ts")

    # Frequency response plot
    plt.figure()
    plt.semilogy(freqs, mag)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('|H(f)|')
    plt.title('Frequency Response')
    plt.grid(True)
    plt.savefig('/home/asg/Music/freqresp.png')
    plt.close()

    # Fit FOPDT model
    L, T, Kss = fit_fopdt(h, Ts0)
    print(f"FOPDT fit: L={L*1000:.2f} ms, T={T*1000:.2f} ms, Kss={Kss:.3f}")

    # Ziegler–Nichols PID Tuning
    Kp, Ti, Td = design_pid_zn(L, T)
    print(f"PID gains: Kp={Kp:.3f}, Ti={Ti:.3f} s, Td={Td:.3f} s")

    # Compute discrete-time gains for C++ PIDController
    ki_cont = Kp / Ti
    kd_cont = Kp * Td
    ki_discrete = ki_cont * Ts
    kd_discrete = kd_cont / Ts

    print("\n// ——— Gains for C++ PIDController ———")
    print(f"// Sampling period Ts = {Ts*1e3:.3f} ms")
    print(f"double Kp_cont = {Kp:.6f};")
    print(f"double ki_discrete = {ki_discrete:.6f};  // (Kp/Ti)*Ts")
    print(f"double kd_discrete = {kd_discrete:.6f};  // (Kp*Td)/Ts")
    print("""
    Eigen::VectorXd kp = Eigen::VectorXd::Constant(N, Kp_cont);
    Eigen::VectorXd ki = Eigen::VectorXd::Constant(N, ki_discrete);
    Eigen::VectorXd kd = Eigen::VectorXd::Constant(N, kd_discrete);
    PIDController pid(kp, ki, kd, lower_limits, upper_limits, set_point);
    """)

    # --- New: plot FOPDT fit overlayed on data ---
    def fopdt_model(t, L, T, K):
        y = K * (1 - np.exp(-(t - L) / T))
        y[t < L] = 0
        return y

    t_fit_s = np.arange(len(h)) * Ts0  # time axis in seconds for model
    h_cumsum = np.cumsum(h) * Ts0      # cumulative integral of impulse response
    h_fit = fopdt_model(t_fit_s, L, T, Kss)

    plt.figure()
    plt.plot(t_fit_s * 1000, h_cumsum, 'k.', label='Measured (cumsum)')
    plt.plot(t_fit_s * 1000, h_fit, 'r-', label=f'FOPDT fit\n$L$={L*1e3:.1f} ms, $T$={T*1e3:.1f} ms')
    plt.xlabel('Time [ms]')
    plt.ylabel('Cumulative Response')
    plt.title('Cumulative Impulse Response and FOPDT Fit')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/asg/Music/fopdt_fit.png')
    plt.close()

if __name__ == '__main__':
    main()
    
# #!/usr/bin/env python3
# import json
# import numpy as np
# import argparse
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# def load_telemetry(json_file, top_fraction):
#     with open(json_file,'r') as f:
#         data = json.load(f)
#     ts    = np.array(data['timestamps_ms'])
#     poke  = np.array(data['poke_values'])>0
#     imgs  = np.array(data['images'])
#     mean_off = imgs[~poke].mean(axis=0)
#     mean_on  = imgs[poke].mean(axis=0)
#     diff     = np.abs(mean_on - mean_off)
#     sel      = np.argsort(diff)[-int(top_fraction*imgs.shape[1]):]
#     sig      = imgs[:,sel].mean(axis=1)
#     return ts, poke, sig

# def average_impulse(sig, poke, window):
#     onsets = np.where(np.diff(np.concatenate(([0],poke.astype(int))))==1)[0]
#     windows=[]
#     for o in onsets:
#         if o+window<=len(sig):
#             windows.append(sig[o:o+window])
#     return np.array(windows).mean(axis=0)

# def fit_fopdt(h, Ts):
#     s = np.cumsum(h)*Ts
#     t = np.arange(len(s))*Ts
#     def fopdt(t,L,T,K):
#         y = K*(1-np.exp(-(t-L)/T))
#         y[t<L] = 0
#         return y
#     p0 = [Ts, 5*Ts, s[-1]]
#     popt,_ = curve_fit(fopdt, t, s, p0=p0, bounds=([0,0,0],[10*Ts,10*Ts,s[-1]*10]))
#     return popt  # L, T, Kss

# def design_pid_zn(L,T):
#     Kp = 1.2*(T/L)
#     Ti = 2*L
#     Td = 0.5*L
#     return Kp, Ti, Td

# def estimate_bandwidth(h, Ts0, oversample):
#     nfft = 512
#     Hf = np.fft.rfft(h, nfft)
#     freqs = np.fft.rfftfreq(nfft, Ts0)
#     mag = np.abs(Hf)
#     H0 = mag[0]
#     thr = H0/np.sqrt(2)
#     idx = np.where((mag<=thr)&(freqs>0))[0]
#     if len(idx)>0:
#         fc = freqs[idx[0]]
#         fs_opt = oversample * fc
#         return fc, fs_opt, freqs, mag
#     else:
#         return None, None, freqs, mag

# def main():
#     parser = argparse.ArgumentParser(description="Estimate TF and tune PID with optimal fs")
#     parser.add_argument("json_file", help="Path to latency_telem.json")
#     parser.add_argument("--top_fraction", type=float, default=0.05)
#     parser.add_argument("--window", type=int, default=50)
#     parser.add_argument("--oversample", type=float, default=15.0,
#                         help="Factor above fc for sampling rate")
#     args = parser.parse_args()

#     ts, poke, sig = load_telemetry(args.json_file, args.top_fraction)
#     dt = np.median(np.diff(ts))
#     Ts0 = dt/1000.0

#     h = average_impulse(sig, poke, args.window)
#     # Impulse plot
#     tvec = np.arange(len(h))*dt
#     plt.figure(); plt.plot(tvec, h, '-o')
#     plt.xlabel('ms'); plt.title('Impulse response')
#     plt.savefig('/home/asg/Music/impulse.png'); plt.close()

#     # Bandwidth and optimal fs
#     fc, fs_opt, freqs, mag = estimate_bandwidth(h, Ts0, args.oversample)
#     if fc:
#         Ts = 1/fs_opt
#         print(f"-3dB bandwidth: {fc:.2f} Hz; suggested fs={fs_opt:.1f} Hz => Ts={Ts*1000:.2f} ms")
#     else:
#         Ts = Ts0
#         print("Could not determine bandwidth; using original Ts")

#     # Frequency response plot
#     plt.figure()
#     plt.semilogy(freqs, mag)
#     plt.xlabel('Hz'); plt.ylabel('|H|'); plt.title('Frequency response')
#     plt.savefig('/home/asg/Music/freqresp.png'); plt.close()

#     # Fit FOPDT model at Ts
#     L, T, Kss = fit_fopdt(h, Ts)
#     print(f"FOPDT fit: L={L:.3f}s, T={T:.3f}s, Kss={Kss:.3f}")

#     # Z-N PID (Ziegler–Nichols)
#     Kp, Ti, Td = design_pid_zn(L, T)
#     print(f"PID gains: Kp={Kp:.3f}, Ti={Ti:.3f}s, Td={Td:.3f}s")

#     # Compute discrete‐time gains for your C++ PIDController
#     # continuous‐time integrator gain = Kp/Ti
#     # continuous‐time differentiator gain = Kp*Td
#     ki_cont    = Kp / Ti
#     kd_cont    = Kp * Td

#     # your C++ code sums e[n], so multiply integrator by Ts
#     ki_discrete = ki_cont * Ts
#     # your C++ code diffs e[n], so divide differentiator by Ts
#     kd_discrete = kd_cont / Ts

#     print("\n// ——— Gains for C++ PIDController ———")
#     print(f"// Sampling period Ts = {Ts*1e3:.3f} ms")
#     print(f"double Kp_cont = {Kp:.6f};")
#     print(f"double ki_discrete = {ki_discrete:.6f};  // = (Kp/Ti)*Ts")
#     print(f"double kd_discrete = {kd_discrete:.6f};  // = (Kp*Td)/Ts")
#     print("""
#     Eigen::VectorXd kp = Eigen::VectorXd::Constant(N, Kp_cont);
#     Eigen::VectorXd ki = Eigen::VectorXd::Constant(N, ki_discrete);
#     Eigen::VectorXd kd = Eigen::VectorXd::Constant(N, kd_discrete);
#     PIDController pid(kp, ki, kd, lower_limits, upper_limits, set_point);
#     """)
        
# if __name__=='__main__':
#     main()