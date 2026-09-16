#!/usr/bin/env python3
""" 
writes a fits file with reference intensity for jesse's RTC
it calculates this as weighted average of theoretical reference
and the 'lucky imaged' measured intensities
the filtered lucky measured images are filtered by percentil 
measured by aggregated signal in annulus around the pupil (scales with Strehl in ZWFS) 

"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import time
from xaosim.shmlib import shm
from baldr_reference.model import generate_references, load_config
from baldr_reference.pupil_fitting import estimate_pupil_center_subpixel



def get_frames(beam_id, n_frames, frame_sleep=0.001):
    shm_path = f"/dev/shm/baldr{beam_id}.im.shm"
    print(f"Opening shared memory: {shm_path}")
    camera = shm(shm_path)

    total = []
    for index in range(n_frames):
        frame = np.asarray(camera.get_data(), dtype=np.float64)

        total.append(frame) 
        if frame_sleep:
            time.sleep(frame_sleep)
        if index == 0 or (index + 1) % max(1, n_frames // 10) == 0:
            print(f"  clear pupil: {index + 1}/{n_frames}")
    return total , shm_path




def get_background(image):
    border = np.concatenate([
        image[0, :],
        image[-1, :],
        image[:, 0],
        image[:, -1],
    ])
    return np.median(border)


def transform_image(image, angle, shift_x, shift_y, scale):
    """Rotate, shift and scale an image about its centre."""

    angle = np.deg2rad(angle)
    cosine = np.cos(angle) / scale
    sine = np.sin(angle) / scale

    matrix = np.array([
        [cosine, -sine],
        [sine, cosine],
    ])

    centre = (np.array(image.shape) - 1) / 2
    shifted_centre = centre + np.array([shift_y, shift_x])
    offset = centre - matrix @ shifted_centre

    return ndimage.affine_transform(
        image,
        matrix,
        offset=offset,
        output_shape=image.shape,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


parser = argparse.ArgumentParser()

parser.add_argument("--output", type=Path, default=None)
#to do: make reference onskyconfigs for each phasemask , then make this default None, and automatically find based on phasemask name
parser.add_argument("--config", type=Path, default='configs/reference_onsky.example.json')
parser.add_argument("--beam_id", type=int, default=1)
parser.add_argument(
    "--phasemask",
    type=str.upper,
    choices=("H1", "H2", "H3", "H4", "H5",
             "J1", "J2", "J3", "J4", "J5"),
    default="H3",
)
# parser.add_argument(
#     "--crop-center",
#     nargs=2,
#     type=float,
#     metavar=("X", "Y"),
#     default=(15,15),
# )
parser.add_argument(
    "--crop-center",
    nargs=2,
    type=float,
    metavar=("X", "Y"),
    default=(15,15),
)
parser.add_argument(
    "--crop-size",
    nargs=2,
    type=int,
    metavar=("HEIGHT", "WIDTH"),
    default=(17, 17),
)

parser.add_argument("--n-clear", type=int, default=100)
parser.add_argument("--n-lucky", type=int, default=1000)

parser.add_argument("--annulus-radius", type=float, default=9)
parser.add_argument("--annulus-thickness", type=float, default=2)

parser.add_argument(
    "--lucky-percent",
    type=float,
    default=10.0,
    help="Percentage of frames retained.",
)

parser.add_argument(
    "--aggregate",
    choices=("mean", "median"),
    default="median",
)

parser.add_argument("--prior-weight", type=float, default=1.0)
parser.add_argument("--measured-weight", type=float, default=1.0)

parser.add_argument("--overwrite", action="store_true")

args = parser.parse_args()

# to do
# add strehl back in to theory pupil
# update ref with clear as in v1 


# -------------------------------------------------------------------------
# Measure clear pupil
# -------------------------------------------------------------------------

#ove_mask("out")
usr = input('mover mask out, the press enter')

clear_frames, _  = get_frames(args.beam_id, args.n_clear)
clear = np.mean(clear_frames, axis=0)

clear_background = get_background(clear)
clear = clear - clear_background

centre_result = estimate_pupil_center_subpixel(clear)

measured_center_x = centre_result["center_x"]
measured_center_y = centre_result["center_y"]

print(
    "Measured pupil centre:",
    measured_center_x,
    measured_center_y,
)


# Simple pupil support for normalization and fitting.
smooth_clear = ndimage.gaussian_filter(
    np.clip(clear, 0, None),
    sigma=0.7,
)

support = smooth_clear > 0.08 * np.max(smooth_clear)
support = ndimage.binary_erosion(support, iterations=1)

if np.count_nonzero(support) < 8:
    raise ValueError("Could not identify the clear pupil.")

clear_normalized = clear / np.sum(clear[support])


# -------------------------------------------------------------------------
# Generate theoretical prior once
# -------------------------------------------------------------------------

config = load_config(args.config)
config["phase_mask"]["name"] = args.phasemask.upper() # update to user input phasemask for theoretical calculation
# Make the theoretical detector image match the measured camera frame.
config.setdefault("detector", {})["crop"] = list(clear.shape)

material_file = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "baldr_reference"
    / "Exposed_Ma-N_1405_optical_constants.txt"
)

theory_clear, theory_masked, wavelengths = generate_references(
    config,
    material_file,
)

if theory_clear.shape != clear.shape:
    raise ValueError(
        "The theoretical and measured images have different shapes: "
        f"{theory_clear.shape} and {clear.shape}."
    )


# -------------------------------------------------------------------------
# Fast detector-plane grid search
# -------------------------------------------------------------------------

# These ranges are deliberately small and hard-coded.
rotation_grid = [-1.0, 0.0, 1.0]
shift_grid = [-1.0, 0.0, 1.0]
scale_grid = [0.99, 1.00, 1.01]

best_cost = np.inf
best_parameters = None

for rotation in rotation_grid:
    for shift_x in shift_grid:
        for shift_y in shift_grid:
            for scale in scale_grid:

                trial = transform_image(
                    theory_clear,
                    rotation,
                    shift_x,
                    shift_y,
                    scale,
                )

                trial_sum = np.sum(trial[support])

                if trial_sum <= 0:
                    continue

                trial = trial / trial_sum

                cost = np.mean(
                    (trial[support] - clear_normalized[support]) ** 2
                )

                if cost < best_cost:
                    best_cost = cost
                    best_parameters = (
                        rotation,
                        shift_x,
                        shift_y,
                        scale,
                    )


if best_parameters is None:
    raise RuntimeError("The theoretical pupil grid search failed.")

rotation, shift_x, shift_y, scale = best_parameters

print("Best rotation:", rotation)
print("Best X shift:", shift_x)
print("Best Y shift:", shift_y)
print("Best scale:", scale)
print("Best cost:", best_cost)


# Apply the same transformation to the theoretical ZWFS reference.
prior = transform_image(
    theory_masked,
    rotation,
    shift_x,
    shift_y,
    scale,
)

prior = prior / np.sum(prior[support])


# -------------------------------------------------------------------------
# Measure ZWFS frames
# -------------------------------------------------------------------------

#move_mask("in")
usr = input('move mask in, then press enter')
zwfs_frames, _ = get_frames(args.beam_id, args.n_lucky)


# Define the lucky-imaging annulus.
rows, columns = np.indices(clear.shape)

radius = np.hypot(
    columns - measured_center_x,
    rows - measured_center_y,
)

inner_radius = args.annulus_radius - args.annulus_thickness / 2
outer_radius = args.annulus_radius + args.annulus_thickness / 2

annulus = (
    (radius >= inner_radius)
    & (radius <= outer_radius)
    & support
)

if np.count_nonzero(annulus) == 0:
    raise ValueError("The selected annulus contains no pupil pixels.")


# -------------------------------------------------------------------------
# Normalize and rank the ZWFS frames
# -------------------------------------------------------------------------

normalized_frames = []
scores = []

for frame in zwfs_frames:

    frame = frame - get_background(frame)
    frame_sum = np.sum(frame[support])

    if frame_sum <= 0:
        continue

    frame = frame / frame_sum

    normalized_frames.append(frame)
    scores.append(np.mean(frame[annulus]))


normalized_frames = np.asarray(normalized_frames)
scores = np.asarray(scores)

if len(normalized_frames) == 0:
    raise RuntimeError("No valid ZWFS frames were acquired.")


# --lucky-percent 10 retains the highest-scoring 10% of frames.
threshold = np.percentile(
    scores,
    100 - args.lucky_percent,
)

lucky_frames = normalized_frames[scores >= threshold]

print(
    "Selected",
    len(lucky_frames),
    "of",
    len(normalized_frames),
    "ZWFS frames.",
)


# -------------------------------------------------------------------------
# Aggregate lucky frames
# -------------------------------------------------------------------------

if args.aggregate == "mean":
    measured_reference = np.mean(lucky_frames, axis=0)
else:
    measured_reference = np.median(lucky_frames, axis=0)

measured_reference = (
    measured_reference
    / np.sum(measured_reference[support])
)


# -------------------------------------------------------------------------
# Combine measured reference and theoretical prior
# -------------------------------------------------------------------------

total_weight = args.prior_weight + args.measured_weight

if total_weight <= 0:
    raise ValueError("The sum of the prior and measured weights must be positive.")

posterior = (
    args.prior_weight * prior
    + args.measured_weight * measured_reference
) / total_weight


# -------------------------------------------------------------------------
# Crop around the requested centre
# -------------------------------------------------------------------------

# crop_center_x = int(round(args.crop_center[0]))
# crop_center_y = int(round(args.crop_center[1]))
if args.crop_center is None:
    crop_center_x = int(round(measured_center_x))
    crop_center_y = int(round(measured_center_y))
else:
    crop_center_x = int(round(args.crop_center[0]))
    crop_center_y = int(round(args.crop_center[1]))
    
crop_height = args.crop_size[0]
crop_width = args.crop_size[1]

output_center_y = (crop_height - 1) // 2
output_center_x = (crop_width - 1) // 2

start_y = crop_center_y - output_center_y
start_x = crop_center_x - output_center_x
end_y = start_y + crop_height
end_x = start_x + crop_width

if (
    start_y < 0
    or start_x < 0
    or end_y > posterior.shape[0]
    or end_x > posterior.shape[1]
):
    raise ValueError(
        f"Requested crop {args.crop_size} does not fit inside "
        f"the measured frame {posterior.shape}."
    )

posterior_sum = np.sum(posterior)
# now crop
posterior = posterior[start_y:end_y, start_x:end_x]


# -------------------------------------------------------------------------
# Normalize and write Jesse RTC format
# -------------------------------------------------------------------------



if not np.isfinite(posterior_sum) or posterior_sum <= 0:
    raise ValueError("The cropped posterior has a non sum.")

# Negative sign retained for the Jesse RTC convention.
rtc_reference = -posterior / posterior_sum
rtc_reference = np.asarray(rtc_reference, dtype=np.float64).reshape(-1)

if args.output is None:
    args.output = Path(f"/usr/local/etc/B{args.beam_id}_meas_offset_{args.beam_id}.fits") #f"output/B{args.beam_id}_{args.phasemask}_meas_offset.fits")
args.output.parent.mkdir(parents=True, exist_ok=True)

fits.PrimaryHDU(rtc_reference).writeto(
    args.output,
    overwrite=args.overwrite,
)

print("Wrote:", args.output.resolve())
print("Output shape:", posterior.shape)
print("Vector length:", rtc_reference.size)
print("Vector sum:", np.sum(rtc_reference))