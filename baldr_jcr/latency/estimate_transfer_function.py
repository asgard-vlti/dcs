#!/usr/bin/env python3
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

def main(json_file, top_fraction=0.05, window_length=50, nfft=512):
    # Load telemetry
    with open(json_file, 'r') as f:
        data = json.load(f)
    timestamps = np.array(data['timestamps_ms'])
    poke_vals  = np.array(data['poke_values'])
    images     = np.array(data['images'])  # shape (n_frames, n_pixels)

    # Select responsive pixels
    poke_on = poke_vals > 0
    mean_off = images[~poke_on].mean(axis=0)
    mean_on  = images[poke_on].mean(axis=0)
    diff = np.abs(mean_on - mean_off)
    n_pixels = images.shape[1]
    n_select = max(1, int(top_fraction * n_pixels))
    idx = np.argsort(diff)[-n_select:]
    ts = images[:, idx].mean(axis=1)

    # Find poke onsets
    poke_on_int = poke_on.astype(int)
    onsets = np.where(np.diff(np.concatenate(([0], poke_on_int))) == 1)[0]

    # Collect impulse response windows
    windows = []
    for onset in onsets:
        if onset + window_length <= len(ts):
            windows.append(ts[onset:onset + window_length])
    if not windows:
        raise RuntimeError("No valid windows found.")
    windows = np.array(windows)

    # Average impulse response
    h = windows.mean(axis=0)

    # Compute sampling frequency
    dt_ms = np.median(np.diff(timestamps))
    fs = 1000.0 / dt_ms  # Hz

    # Plot impulse response
    plt.figure(figsize=(6,4))
    times = np.arange(window_length) * dt_ms
    plt.plot(times, h, '-o')
    plt.xlabel('Time since poke (ms)')
    plt.ylabel('Response amplitude')
    plt.title('Average Impulse Response')
    plt.tight_layout()
    plt.savefig('/home/asg/Music/impulse_response.png')
    plt.close()
    print("Saved impulse response to impulse_response.png")

    # Compute frequency response via FFT
    H = np.fft.rfft(h, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=dt_ms/1000.0)

    # Plot frequency response
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6))
    ax1.semilogy(freqs, np.abs(H))
    ax1.set_ylabel('|H(f)|')
    ax1.set_title('Estimated Frequency Response')
    ax1.grid(True, which='both', ls='--', alpha=0.5)

    ax2.plot(freqs, np.unwrap(np.angle(H)))
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (rad)')
    ax2.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('/home/asg/Music/frequency_response.png')
    plt.close(fig)
    print("Saved frequency response to frequency_response.png")

    # Suppose Hf and freqs are from:
#   Hf = np.fft.rfft(h, n=nfft)
#   freqs = np.fft.rfftfreq(nfft, d=Ts)

    # Find DC magnitude for reference:
    H0 = np.abs(H[0])

    # Locate f_c where |H(f_c)| = H0 / sqrt(2)  (–3 dB point)
    mag = np.abs(H)
    threshold = H0 / np.sqrt(2)
    # Only consider freqs > 0
    idx = np.where((mag <= threshold) & (freqs > 0))[0]
    if len(idx) > 0:
        fc = freqs[idx[0]]
        print(f"–3 dB bandwidth: {fc:.1f} Hz")
        # Recommended sample rate
        fs_opt = 15 * fc  # you can choose any factor 10–20
        print(f"Suggested sample rate: {fs_opt:.0f} Hz (≈{1/fs_opt*1e3:.2f} ms/sample)")
    else:
        print("Could not find a –3 dB point; your system may be very slow or non-minimum phase.")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate TF from average impulse response")
    parser.add_argument("json_file", help="Path to latency_telem.json")
    parser.add_argument("--top_fraction", type=float, default=0.05,
                        help="Fraction of pixels to select")
    parser.add_argument("--window_length", type=int, default=50,
                        help="Number of frames to include in impulse response")
    parser.add_argument("--nfft", type=int, default=512,
                        help="FFT length for frequency response")
    args = parser.parse_args()
    main(args.json_file, args.top_fraction, args.window_length, args.nfft)