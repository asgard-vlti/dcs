#!/usr/bin/env python
import json
import numpy as np
import argparse
import matplotlib.pyplot as plt

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    timestamps = np.array(data['timestamps_ms'])
    poke_vals  = np.array(data['poke_values'])
    images     = np.array(data['images'])
    return timestamps, poke_vals, images

def select_pixels(images, poke_vals, top_fraction):
    poke_on    = poke_vals > 0
    mean_off   = images[~poke_on].mean(axis=0)
    mean_on    = images[poke_on].mean(axis=0)
    diff       = np.abs(mean_on - mean_off)
    n_pixels   = images.shape[1]
    n_select   = max(1, int(top_fraction * n_pixels))
    selected_idx = np.argsort(diff)[-n_select:]
    return selected_idx, mean_on

def plot_time_series(timestamps, images, selected_idx, poke_vals, output):
    ts = images[:, selected_idx].mean(axis=1)
    poke_on = poke_vals > 0
    onsets  = np.where(np.diff(np.concatenate(([0], poke_on.astype(int)))) == 1)[0]
    poke_times = timestamps[onsets]

    plt.figure(figsize=(8,4))
    plt.plot(timestamps, ts, label='Mean selected pixels')
    for t in poke_times:
        plt.axvline(t, color='r', linestyle='--', alpha=0.6)
    plt.xlabel('Time (ms)')
    plt.ylabel('Mean intensity')
    plt.title('Time Series of Selected Pixels')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    print(f"Saved time series plot to {output}")

def plot_overlay(mean_on, selected_idx, side, output):
    plt.figure(figsize=(6,6))
    plt.imshow(mean_on.reshape(side, side), cmap='gray')
    ys = selected_idx // side
    xs = selected_idx % side
    plt.scatter(xs, ys, marker='x', color='cyan', s=20, label='Selected pixels')
    plt.title('Mean ON Image with Selected Pixels')
    plt.axis('off')
    plt.legend(loc='lower right')
    plt.savefig(output)
    plt.close()
    print(f"Saved overlay plot to {output}")


def plot_detailed_latency(images, timestamps, poke_vals, selected_idx, max_offset, output):
    ts = images[:, selected_idx].mean(axis=1)
    poke_on = poke_vals > 0
    onsets  = np.where(np.diff(np.concatenate(([0], poke_on.astype(int)))) == 1)[0]
    offsets = np.arange(-1, max_offset+1)

    # Build data matrix
    data_matrix = []
    for onset in onsets:
        if onset - 1 < 0 or onset + max_offset >= len(ts):
            continue
        data_matrix.append(ts[onset + offsets])
    data_matrix = np.array(data_matrix)
    mean_trace = data_matrix.mean(axis=0)
    std_trace  = data_matrix.std(axis=0)

    # Compute median dt (ms per frame)
    dt = np.median(np.diff(timestamps))

    # Time axis values
    time_offsets = offsets * dt

    # Create figure
    fig, ax_time = plt.subplots(figsize=(6,4))

    # Plot individual and mean±std traces vs time
    for trace in data_matrix:
        ax_time.plot(time_offsets, trace, color='gray', alpha=0.3)
    ax_time.errorbar(time_offsets, mean_trace, yerr=std_trace,
                     fmt='-o', color='blue', label='Mean ± std')
    ax_time.axvline(0, color='red', linestyle='--', label='Poke frame')

    ax_time.set_xlabel('Time since poke (ms)')
    ax_time.set_ylabel('Mean intensity')
    ax_time.set_title('Latency Trace Around Poke')

    # Create top axis for frame offset
    ax_frames = ax_time.twiny()
    ax_frames.set_xlim(ax_time.get_xlim())

    # Use the same tick positions for both axes
    ax_time.set_xticks(time_offsets)
    ax_frames.set_xticks(time_offsets)

    # Label bottom with time, top with frame offset
    ax_time.set_xticklabels([f"{t:.1f}" for t in time_offsets])
    ax_frames.set_xticklabels(offsets)

    ax_frames.set_xlabel('Frame offset')

    # Legend on the time axis
    ax_time.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved detailed latency plot to {output}")

    
# def plot_detailed_latency(images, timestamps, poke_vals, selected_idx, max_offset, output):
#     ts = images[:, selected_idx].mean(axis=1)
#     poke_on = poke_vals > 0
#     onsets  = np.where(np.diff(np.concatenate(([0], poke_on.astype(int)))) == 1)[0]
#     offsets = np.arange(-1, max_offset+1)

#     data_matrix = []
#     for onset in onsets:
#         if onset - 1 < 0 or onset + max_offset >= len(ts):
#             continue
#         data_matrix.append(ts[onset + offsets])
#     data_matrix = np.array(data_matrix)

#     mean_trace = data_matrix.mean(axis=0)
#     std_trace  = data_matrix.std(axis=0)

#     # Estimate latency at half-max crossing
#     baseline = mean_trace[0]
#     peak     = mean_trace.max()
#     thresh   = (baseline + peak) / 2
#     idx      = np.where(mean_trace > thresh)[0]
#     est_lat  = offsets[idx[0]] if idx.size else None

#     print("Offset\tMean\tStd")
#     for off, m, s in zip(offsets, mean_trace, std_trace):
#         print(f"{off:3d}\t{m:.3f}\t{s:.3f}")
#     if est_lat is not None:
#         print(f"\nEstimated latency: {est_lat} frames")

#     plt.figure(figsize=(6,4))
#     for trace in data_matrix:
#         plt.plot(offsets, trace, color='gray', alpha=0.3)
#     plt.errorbar(offsets, mean_trace, yerr=std_trace, fmt='-o',
#                  color='blue', label='Mean ± std')
#     plt.axvline(0, color='red', linestyle='--', label='Poke frame')
#     plt.xlabel('Frame offset')
#     plt.ylabel('Mean intensity')
#     plt.title('Latency Trace Around Poke')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(output)
#     plt.close()
#     print(f"Saved detailed latency plot to {output}")

def main():
    parser = argparse.ArgumentParser(description="Combined latency analysis")
    parser.add_argument("json_file", help="Path to latency_telem.json")
    parser.add_argument("--top_fraction", type=float, default=0.05,
                        help="Fraction of pixels to select")
    parser.add_argument("--max_offset", type=int, default=5,
                        help="Frames after poke to analyze")
    args = parser.parse_args()

    timestamps, poke_vals, images = load_data(args.json_file)
    selected_idx, mean_on = select_pixels(images, poke_vals, args.top_fraction)
    side = int(np.sqrt(images.shape[1]))

    plot_time_series(timestamps, images, selected_idx, poke_vals,
                     '/home/asg/Music/combined_timeseries.png')
    plot_overlay(mean_on, selected_idx, side,
                 '/home/asg/Music/combined_overlay.png')
    plot_detailed_latency(images, timestamps, poke_vals, selected_idx,
                          args.max_offset, '/home/asg/Music/combined_latency_trace.png')

if __name__ == "__main__":
    main()

# #!/usr/bin/env python
# import json
# import numpy as np
# import argparse
# import matplotlib.pyplot as plt

# def analyze_selected_pixels_with_overlay(json_file, top_fraction=0.05):
#     # Load JSON
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#     timestamps = np.array(data['timestamps_ms'])
#     poke_vals = np.array(data['poke_values'])
#     images = np.array(data['images'])  # shape (n_frames, n_pixels)

#     n_frames, n_pixels = images.shape
#     side = int(np.round(np.sqrt(n_pixels)))
#     if side * side != n_pixels:
#         raise ValueError("Image is not square")

#     # Identify ON vs OFF frames
#     poke_on = poke_vals > 0
#     poke_off = ~poke_on

#     # Compute per-pixel mean for OFF and ON
#     mean_off = images[poke_off].mean(axis=0)
#     mean_on  = images[poke_on].mean(axis=0)

#     # Compute absolute difference per pixel
#     diff = np.abs(mean_on - mean_off)

#     # Select top fraction of pixels
#     n_select = max(1, int(top_fraction * n_pixels))
#     selected_idx = np.argsort(diff)[-n_select:]

#     # Compute time series of mean over selected pixels
#     selected_ts = images[:, selected_idx].mean(axis=1)

#     # Identify poke onsets for marking
#     onsets = np.where(np.diff(np.concatenate(([0], poke_on.astype(int)))) == 1)[0]
#     poke_times = timestamps[onsets]

#     # Plot 1: time series
#     plt.figure(figsize=(8,4))
#     plt.plot(timestamps, selected_ts, label=f'Mean of top {n_select} pixels')
#     for t in poke_times:
#         plt.axvline(t, color='r', linestyle='--', alpha=0.6)
#     plt.xlabel('Time (ms)')
#     plt.ylabel('Mean intensity of selected pixels')
#     plt.title('Latency Test: Selected Pixel Response & Poke Onsets')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('selected_pixels_latency_ts.png')
#     plt.close()
#     print(f"Saved time series plot to selected_pixels_latency_ts.png with {n_select} pixels.")

#     # Plot 2: overlay on mean ON image
#     mean_on_img = mean_on.reshape(side, side)
#     plt.figure(figsize=(6,6))
#     plt.imshow(mean_on_img, cmap='gray')
#     # compute x,y coords of selected_idx
#     ys = selected_idx // side
#     xs = selected_idx % side
#     plt.scatter(xs, ys, marker='x', color='cyan', s=20, label='Selected pixels')
#     plt.title('Mean ON Image with Selected Pixels')
#     plt.axis('off')
#     plt.legend(loc='lower right')
#     plt.savefig('selected_pixels_overlay.png')
#     plt.close()
#     print("Saved overlay plot to selected_pixels_overlay.png")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Analyze latency and overlay selected pixels"
#     )
#     parser.add_argument(
#         "json_file",
#         help="Path to latency_telem.json"
#     )
#     parser.add_argument(
#         "--top_fraction",
#         type=float,
#         default=0.05,
#         help="Fraction of pixels to select (default: 0.05)"
#     )
#     args = parser.parse_args()
#     analyze_selected_pixels_with_overlay(args.json_file, args.top_fraction)

# # import json
# # import numpy as np
# # import argparse
# # import matplotlib.pyplot as plt

# # def analyze_selected_pixels(json_file, top_fraction=0.05):
# #     # Load JSON
# #     with open(json_file, 'r') as f:
# #         data = json.load(f)
# #     timestamps = np.array(data['timestamps_ms'])
# #     poke_vals = np.array(data['poke_values'])
# #     images = np.array(data['images'])  # shape (n_frames, n_pixels)

# #     n_frames, n_pixels = images.shape

# #     # Identify ON vs OFF frames
# #     poke_on = poke_vals > 0
# #     poke_off = ~poke_on

# #     # Compute per-pixel mean for ON and OFF
# #     mean_off = images[poke_off].mean(axis=0)
# #     mean_on  = images[poke_on].mean(axis=0)

# #     # Compute absolute difference per pixel
# #     diff = np.abs(mean_on - mean_off)

# #     # Select top fraction of pixels
# #     n_select = max(1, int(top_fraction * n_pixels))
# #     selected_idx = np.argsort(diff)[-n_select:]

# #     # Compute time series of mean over selected pixels
# #     selected_ts = images[:, selected_idx].mean(axis=1)

# #     # Identify poke onsets for marking
# #     onsets = np.where(np.diff(np.concatenate(([0], poke_on.astype(int)))) == 1)[0]
# #     poke_times = timestamps[onsets]

# #     # Plot time series
# #     plt.figure(figsize=(8,4))
# #     plt.plot(timestamps, selected_ts, label=f'Mean of top {n_select} pixels')
# #     for t in poke_times:
# #         plt.axvline(t, color='r', linestyle='--', alpha=0.6)
# #     plt.xlabel('Time (ms)')
# #     plt.ylabel('Mean intensity of selected pixels')
# #     plt.title('Latency Test: Selected Pixel Response & Poke Onsets')
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig('delme.png')
# #     print(f"Saved plot to delme.png with {n_select} pixels selected.")

# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(description="Analyze latency using selected high-difference pixels")
# #     parser.add_argument("json_file", help="Path to latency_telem.json")
# #     parser.add_argument("--top_fraction", type=float, default=0.05,
# #                         help="Fraction of pixels to select (default: 0.05)")
# #     args = parser.parse_args()
# #     analyze_selected_pixels(args.json_file, args.top_fraction)
# # # import json
# # # import numpy as np
# # # import argparse
# # # import matplotlib.pyplot as plt

# # # def analyze_latency_with_pupil(json_file):
# # #     # Load JSON
# # #     with open(json_file, 'r') as f:
# # #         data = json.load(f)
# # #     timestamps = np.array(data['timestamps_ms'])
# # #     poke_vals = np.array(data['poke_values'])
# # #     images = np.array(data['images'])  # shape (n_frames, n_pixels)

# # #     # Determine image dimensions (assumes square)
# # #     n_frames, n_pixels = images.shape
# # #     side = int(np.round(np.sqrt(n_pixels)))
# # #     if side * side != n_pixels:
# # #         raise ValueError("Image is not square")
    
# # #     # Build circular pupil mask (centered, radius = 0.45*side)
# # #     yy, xx = np.meshgrid(np.arange(side), np.arange(side))
# # #     cx = (side - 1) / 2.0
# # #     cy = (side - 1) / 2.0
# # #     r = side * 0.45
# # #     mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
# # #     mask_flat = mask.flatten()

# # #     # Compute mean intensity inside pupil
# # #     pupil_means = np.array([frame[mask_flat].mean() for frame in images])

# # #     # Identify poke onsets (transition from 0 to >0)
# # #     poke_on = (poke_vals > 0).astype(int)
# # #     onsets = np.where(np.diff(np.concatenate(([0], poke_on))) == 1)[0]
# # #     poke_times = timestamps[onsets]

# # #     # Plot time series
# # #     plt.figure(figsize=(8,4))
# # #     plt.plot(timestamps, pupil_means, label='Mean pupil intensity')
# # #     for t in poke_times:
# # #         plt.axvline(t, color='r', linestyle='--', alpha=0.6)
# # #     plt.xlabel('Time (ms)')
# # #     plt.ylabel('Mean intensity in pupil')
# # #     plt.title('Latency Test: Pupil Intensity & DM Poke Onsets')
# # #     plt.legend()
# # #     plt.tight_layout()
# # #     plt.savefig('delme.png')
# # #     print("Saved plot to delme.png")
# # #     plt.close()

# # # if __name__ == "__main__":
# # #     parser = argparse.ArgumentParser(description="Analyze latency and plot pupil intensity")
# # #     parser.add_argument("json_file", help="Path to latency_telem.json")
# # #     args = parser.parse_args()
# # #     analyze_latency_with_pupil(args.json_file)

# # # # # Write the analyze_latency.py script to a file for the user
# # # # #!/usr/bin/env python
# # # # import json
# # # # import numpy as np
# # # # import argparse
# # # # import matplotlib.pyplot as plt

# # # # def analyze_latency(json_file, threshold_factor=3, pre_poke_window=5):
# # # #     # Load JSON
# # # #     with open(json_file, 'r') as f:
# # # #         data = json.load(f)
# # # #     timestamps = np.array(data['timestamps_ms'])
# # # #     poke_vals = np.array(data['poke_values'])
# # # #     images = np.array(data['images'])  # shape (n_frames, n_pixels)

# # # #     # Compute mean intensity per frame
# # # #     mean_intensity = images.mean(axis=1)

# # # #     # Identify poke ON phases: when poke_vals == poke_amp
# # # #     poke_on = np.where((poke_vals > 0))[0]

# # # #     # Find the start of each ON block
# # # #     starts = poke_on[np.where(np.diff(np.concatenate(([0], poke_on))) > 1)]

# # # #     delays = []
# # # #     for onset in starts:
# # # #         # Define baseline from pre_poke_window before onset
# # # #         start = max(0, onset - pre_poke_window)
# # # #         baseline = mean_intensity[start:onset]
# # # #         mu, sigma = baseline.mean(), baseline.std()
# # # #         threshold = mu + threshold_factor * sigma

# # # #         # Search for first frame after onset exceeding threshold
# # # #         post = mean_intensity[onset:]
# # # #         above = np.where(post > threshold)[0]
# # # #         if above.size > 0:
# # # #             delay = above[0]
# # # #             delays.append(delay)

# # # #     delays = np.array(delays)
# # # #     print(f"Detected {len(delays)} poke events.")
# # # #     if len(delays) == 0:
# # # #         print("No events exceeded threshold.")
# # # #         return

# # # #     # Output statistics
# # # #     dt = np.mean(np.diff(timestamps))  # average ms per frame
# # # #     print(f"Delays (frames): mean={delays.mean():.2f}, std={delays.std():.2f}")
# # # #     print(f"Delays (ms): mean={delays.mean()*dt:.2f}, std={delays.std()*dt:.2f}")

# # # #     # Plot histogram
# # # #     plt.figure()
# # # #     plt.hist(delays, bins=range(0, int(delays.max())+2), align='left', rwidth=0.8)
# # # #     plt.xlabel('Delay (frames)')
# # # #     plt.ylabel('Count')
# # # #     plt.title('Latency Histogram')
# # # #     plt.savefig('delme.png')
# # # #     plt.close()


# # # # #analyze_latency(args.json_file, args.threshold_factor, args.pre_poke_window)

# # # # if __name__ == "__main__":
# # # #     parser = argparse.ArgumentParser(description="Analyze latency from latency_telem.json")
# # # #     parser.add_argument("json_file", help="Path to latency_telem.json")
# # # #     parser.add_argument("--threshold_factor", type=float, default=3.0, 
# # # #                         help="Number of baseline std devs for detection (default: 3)")
# # # #     parser.add_argument("--pre_poke_window", type=int, default=5,
# # # #                         help="Number of frames before poke to compute baseline (default: 5)")
# # # #     args = parser.parse_args()

# # # #     analyze_latency(args.json_file, args.threshold_factor, args.pre_poke_window)
