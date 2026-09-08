"""Minimal Baldr clear-pupil and phase-mask reference model.

This module is independent of BaldrApp.  Its equations and array conventions
are extracted from the BaldrApp Fresnel/polychromatic reference path so that a
reference product can be generated with NumPy, SciPy, and Astropy alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.interpolate import interp1d


PHASE_MASK_PRESETS = {
    "J1": {"depth_um": 0.474, "diameter_um": 32.0},
    "J2": {"depth_um": 0.474, "diameter_um": 36.0},
    "J3": {"depth_um": 0.474, "diameter_um": 44.0},
    "J4": {"depth_um": 0.474, "diameter_um": 54.0},
    "J5": {"depth_um": 0.474, "diameter_um": 65.0},
    "H1": {"depth_um": 0.654, "diameter_um": 31.0},
    "H2": {"depth_um": 0.654, "diameter_um": 37.0},
    "H3": {"depth_um": 0.654, "diameter_um": 44.0},
    "H4": {"depth_um": 0.654, "diameter_um": 53.0},
    "H5": {"depth_um": 0.654, "diameter_um": 68.0},
}


SPECTRAL_TYPE_TEMPERATURE_K = {
    "O5V": 40000.0,
    "B0V": 30000.0,
    "B5V": 15000.0,
    "A0V": 9500.0,
    "A5V": 8200.0,
    "F0V": 7300.0,
    "F5V": 6500.0,
    "G0V": 5950.0,
    "G2V": 5770.0,
    "G5V": 5600.0,
    "K0V": 5250.0,
    "K5V": 4400.0,
    "M0V": 3850.0,
    "M2V": 3500.0,
    "M5V": 3150.0,
}


def _disc(dim: int, diameter: float, obstruction: float = 0.0) -> np.ndarray:
    center = (dim - 1) / 2.0
    yy, xx = np.indices((dim, dim), dtype=float)
    radius = np.hypot(xx - center, yy - center)
    outer = radius <= diameter / 2.0
    if obstruction <= 0:
        return outer.astype(float)
    inner = radius <= diameter * obstruction / 2.0
    return (outer & ~inner).astype(float)


def _rotate_about(array: np.ndarray, angle_deg: float, center) -> np.ndarray:
    angle = -np.deg2rad(angle_deg)
    y, x = np.indices(array.shape, dtype=float)
    xp = (x - center[0]) * np.cos(angle) + (y - center[1]) * np.sin(angle) + center[0]
    yp = -(x - center[0]) * np.sin(angle) + (y - center[1]) * np.cos(angle) + center[1]
    return ndimage.map_coordinates(array, [yp, xp], mode="constant", cval=0, order=3)


def _telescope_pupil(dim: int, diameter: float, geometry: str) -> np.ndarray:
    if geometry == "ut":
        obstruction, spider_tilt = 1100 / 8000, 5.5
    elif geometry == "at":
        obstruction, spider_tilt = 0.13 / 1.8, 2.5
    else:
        raise ValueError("Telescope pupil must be UT or AT.")

    padded_size = dim + 50
    center = padded_size // 2
    thickness = int(max(1, 0.008 * dim))
    spiders = []
    reference = np.zeros((padded_size, padded_size))
    reference[center:, center:center + thickness] = 1
    spiders.append(_rotate_about(reference, -spider_tilt, (center, center + diameter / 2)))
    reference = np.zeros((padded_size, padded_size))
    reference[:center, center - thickness + 1:center + 1] = 1
    spiders.append(_rotate_about(reference, -spider_tilt, (center, center - diameter / 2)))
    reference = np.zeros((padded_size, padded_size))
    reference[center:center + thickness, center:] = 1
    spiders.append(_rotate_about(reference, spider_tilt, (center + diameter / 2, center)))
    reference = np.zeros((padded_size, padded_size))
    reference[center - thickness + 1:center + 1, :center] = 1
    spiders.append(_rotate_about(reference, spider_tilt, (center - diameter / 2, center)))
    spider_transmission = 1 - _rotate_about(sum(spiders), 45, (center, center))
    spider_transmission = spider_transmission[25:-25, 25:-25]
    return (_disc(dim, diameter, obstruction) * spider_transmission >= 0.5).astype(float)


def _custom_pupil(dim: int, diameter: float, pupil: dict) -> np.ndarray:
    obstruction = float(pupil.get("central_obstruction", 0.0))
    if not 0.0 <= obstruction < 1.0:
        raise ValueError("pupil.central_obstruction must lie in [0, 1).")
    result = _disc(dim, diameter, obstruction)
    spiders = pupil.get("spiders")
    if spiders is None:
        return result
    if "orientation_deg" in spiders:
        raise ValueError(
            "pupil.spiders.orientation_deg is no longer supported; use "
            "pupil.rotation_deg to rotate the complete pupil."
        )

    count = int(spiders.get("count", 0))
    thickness = float(spiders.get("thickness", 0.0))
    if count < 0:
        raise ValueError("pupil.spiders.count must be non-negative.")
    if thickness < 0:
        raise ValueError("pupil.spiders.thickness must be non-negative.")
    if count == 0 or thickness == 0:
        return result

    center = (dim - 1) / 2.0
    y, x = np.indices((dim, dim), dtype=float)
    x -= center
    y -= center
    inner_radius = obstruction * diameter / 2
    outer_radius = diameter / 2
    half_width = thickness * diameter / 2
    spider_mask = np.zeros((dim, dim), dtype=bool)
    for index in range(count):
        angle = np.deg2rad(index * 360.0 / count)
        along = x * np.cos(angle) + y * np.sin(angle)
        across = -x * np.sin(angle) + y * np.cos(angle)
        spider_mask |= (
            (along >= inner_radius)
            & (along <= outer_radius)
            & (np.abs(across) <= half_width)
        )
    result[spider_mask] = 0.0
    return result


def make_pupil(config: dict) -> np.ndarray:
    pupil = config["pupil"]
    dim = int(pupil["array_size"])
    diameter = float(pupil["pixels_across_pupil"])
    geometry_value = pupil.get("geometry")
    if geometry_value is None or not str(geometry_value).strip():
        result = _custom_pupil(dim, diameter, pupil)
    else:
        geometry = str(geometry_value).strip().lower()
        if geometry == "solarstein":
            result = _disc(dim, diameter, 1100 / 8000)
        elif geometry in {"disc", "disk"}:
            result = _disc(dim, diameter)
        elif geometry in {"ut", "at"}:
            result = _telescope_pupil(dim, diameter, geometry)
        else:
            raise ValueError(
                "pupil.geometry must be 'solarstein', 'disc', 'UT', 'AT', "
                "or omitted for a custom pupil."
            )
    rotation = float(pupil.get("rotation_deg", 0.0))
    if rotation:
        center = ((dim - 1) / 2, (dim - 1) / 2)
        result = (_rotate_about(result, rotation, center) >= 0.5).astype(float)
    return result


def _planck(wavelength_m: np.ndarray, temperature_k: float) -> np.ndarray:
    h = 6.62607015e-34
    c = 299792458.0
    k = 1.380649e-23
    exponent = h * c / (wavelength_m * k * temperature_k)
    return (2 * h * c**2) / wavelength_m**5 / np.expm1(exponent)


def resolve_source_profile(config: dict) -> tuple[str, dict, float, str | None]:
    """Resolve the selected profile and its effective blackbody temperature.

    A non-null ``temperature_K`` always wins. Otherwise ``spectral_type`` is
    looked up in ``SPECTRAL_TYPE_TEMPERATURE_K``.
    """
    source = config["source"]
    profile_name = str(source["profile"])
    try:
        profile = source["profiles"][profile_name]
    except KeyError as error:
        raise ValueError(f"Unknown source profile {profile_name!r}.") from error

    temperature = profile.get("temperature_K")
    spectral_type = profile.get("spectral_type")
    if temperature is not None:
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("source profile temperature_K must be finite and positive.")
        return profile_name, profile, temperature, spectral_type

    if spectral_type is None or not str(spectral_type).strip():
        raise ValueError(
            f"Source profile {profile_name!r} must provide temperature_K or spectral_type."
        )
    spectral_type = str(spectral_type).strip().upper().replace(" ", "")
    try:
        temperature = SPECTRAL_TYPE_TEMPERATURE_K[spectral_type]
    except KeyError as error:
        raise ValueError(
            f"Unknown spectral type {spectral_type!r}; supported types are "
            f"{', '.join(SPECTRAL_TYPE_TEMPERATURE_K)}."
        ) from error
    return profile_name, profile, temperature, spectral_type


def make_spectrum(config: dict) -> tuple[np.ndarray, np.ndarray]:
    _, _, temperature_k, _ = resolve_source_profile(config)
    spectrum = config["spectrum"]
    wavelength_min_m = float(spectrum["wavelength_min_m"])
    wavelength_max_m = float(spectrum["wavelength_max_m"])
    if wavelength_max_m <= wavelength_min_m:
        raise ValueError("spectrum.wavelength_max_m must exceed wavelength_min_m.")
    wavelengths = np.linspace(
        wavelength_min_m, wavelength_max_m, int(spectrum["samples"])
    )
    mode = str(spectrum.get("mode", "blackbody")).lower()
    if mode == "flat":
        weights = np.ones_like(wavelengths)
    elif mode == "blackbody":
        weights = _planck(wavelengths, temperature_k)
        if str(spectrum.get("weighting", "photon")).lower() == "photon":
            weights *= wavelengths
    else:
        raise ValueError("spectrum.mode must be 'blackbody' or 'flat'.")
    weights /= weights.sum()
    bandwidth_nm = (wavelength_max_m - wavelength_min_m) * 1e9
    return wavelengths, weights * bandwidth_nm


def _linear_interpolator(data_path: Path):
    table = np.loadtxt(data_path, skiprows=2, usecols=(0, 1))
    return interp1d(
        table[:, 0], table[:, 1], kind="linear", bounds_error=False,
        fill_value="extrapolate",
    )


def mask_parameters(
    config: dict, wavelength_m: float, material_data_path: Path
) -> tuple[float, float]:
    mask = dict(config["phase_mask"])
    mask_name = mask.get("name")
    if mask_name is not None and str(mask_name).strip():
        mask_name = str(mask_name).strip().upper()
        if mask_name not in PHASE_MASK_PRESETS:
            raise ValueError(
                f"Unknown phase-mask name {mask_name!r}; use H1-H5, J1-J5, "
                "or null for custom physical parameters."
            )
        mask.update(PHASE_MASK_PRESETS[mask_name])
        mask["phase_model"] = "physical_depth"
        mask["diameter_model"] = "physical"
        mask.setdefault("material", "N_1405")
    theta_model = str(mask.get("phase_model", "physical_depth")).lower()
    if theta_model == "constant":
        theta = float(mask["phase_shift_rad"])
    elif theta_model == "physical_depth":
        wavelength_um = wavelength_m * 1e6
        refractive_index = float(_linear_interpolator(material_data_path)(wavelength_um * 1e3))
        theta = (
            2 * np.pi / wavelength_um
            * float(mask["depth_um"])
            * (refractive_index - 1)
        )
    else:
        raise ValueError("phase_mask.phase_model must be 'constant' or 'physical_depth'.")

    diameter_model = str(mask.get("diameter_model", "physical")).lower()
    if diameter_model == "physical":
        diameter_lambda_d = (
            float(mask["diameter_um"]) * 1e-6
            / (1.22 * float(config["optics"]["f_number"]) * wavelength_m)
        )
    elif diameter_model == "lambda_over_d":
        diameter_lambda_d = float(mask["diameter_lambda_over_d"])
    else:
        raise ValueError("phase_mask.diameter_model must be 'physical' or 'lambda_over_d'.")
    return theta, diameter_lambda_d


def zwfs_field(
    amplitude: np.ndarray, theta: float, diameter_lambda_d: float,
    focal_array_size: int = 300, phase: np.ndarray | None = None, strehl: float = 1.0
) -> np.ndarray:
    """Return the post-phase-mask pupil field using BaldrApp centering."""
    original_size = amplitude.shape[0]
    if amplitude.shape != (original_size, original_size):
        raise ValueError("The input amplitude must be square.")
    entrance = amplitude.astype(complex)
    if phase is not None:
        entrance *= np.exp(1j * np.asarray(phase, dtype=float))
    if original_size % 2 == 0:
        work = np.pad(entrance, ((0, 1), (0, 1)))
    else:
        work = entrance
    size = work.shape[0]
    focal_size = max(int(focal_array_size), size)
    if focal_size % 2 == 0:
        focal_size += 1
    pad = (focal_size - size) // 2
    pupil_field = np.zeros((focal_size, focal_size), dtype=complex)
    pupil_field[pad:pad + size, pad:pad + size] = work
    focal_field = np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(pupil_field), norm="ortho")
    )

    support = np.zeros((focal_size, focal_size), dtype=bool)
    support[pad:pad + size, pad:pad + size] = np.abs(work) > 0
    yy, xx = np.indices((focal_size, focal_size))
    center = (focal_size - 1) / 2.0
    radius = np.hypot(yy - center, xx - center)
    pupil_diameter_pixels = 2.0 * radius[support].max()
    pixels_per_lambda_d = focal_size / pupil_diameter_pixels
    phase_disc = radius <= 0.5 * diameter_lambda_d * pixels_per_lambda_d

    # reference wave = b in N'Diaye notation which scales with sqrt strehl ratio in high strehl regimes
    reference_wave = np.sqrt( strehl ) * np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(phase_disc * focal_field), norm="ortho")
    )
    output = pupil_field + (np.exp(1j * theta) - 1.0) * reference_wave
    output = output[pad:pad + size, pad:pad + size]
    return output[:original_size, :original_size]


def _coordinate_grid(shape: tuple[int, int], dx: float, dy: float | None = None):
    if dy is None:
        dy = dx
    ny, nx = shape
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    return np.meshgrid(x, y)


def _angular_spectrum(field, wavelength, dx, z, dy=None):
    if dy is None:
        dy = dx
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    argument = 1.0 - (wavelength * fx_grid) ** 2 - (wavelength * fy_grid) ** 2
    valid = argument >= 0
    transfer = np.zeros(field.shape, dtype=complex)
    transfer[valid] = np.exp(1j * 2 * np.pi / wavelength * z * np.sqrt(argument[valid]))
    return np.fft.ifft2(np.fft.fft2(field) * transfer)


def _thin_lens(field, wavelength, dx, focal_length):
    x, y = _coordinate_grid(field.shape, dx)
    return field * np.exp(-1j * np.pi * (x**2 + y**2) / (wavelength * focal_length))


def _one_step_fresnel(field, wavelength, dx, z, dy=None):
    if dy is None:
        dy = dx
    ny, nx = field.shape
    k = 2 * np.pi / wavelength
    x1, y1 = _coordinate_grid(field.shape, dx, dy)
    dx_out = wavelength * abs(z) / (nx * dx)
    dy_out = wavelength * abs(z) / (ny * dy)
    x2, y2 = _coordinate_grid(field.shape, dx_out, dy_out)
    quad_in = np.exp(1j * k * (x1**2 + y1**2) / (2 * z))
    quad_out = np.exp(1j * k * (x2**2 + y2**2) / (2 * z))
    transformed = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field * quad_in)))
    output = (dx * dy) / (1j * wavelength * z) * quad_out * transformed
    return output, dx_out, dy_out


def derive_relay(config: dict) -> dict:
    relay = dict(config["relay"])
    diameter = relay["entrance_diameter_m"] * relay["collimator_focal_length_m"] / relay["oap_focal_length_m"]
    magnification = relay["detector_pupil_diameter_m"] / diameter
    object_nominal = relay["imaging_focal_length_m"] * (1 + 1 / magnification)
    image_nominal = relay["imaging_focal_length_m"] * (1 + magnification)
    mirror_to_lens = object_nominal - relay["distance_to_mirror_m"]
    object_distance = relay["distance_to_mirror_m"] + mirror_to_lens
    image_distance = 1 / (1 / relay["imaging_focal_length_m"] - 1 / object_distance)
    relay.update(
        physical_pupil_diameter_m=diameter,
        mirror_to_lens_m=mirror_to_lens,
        focus_to_detector_m=(
            image_distance - relay["imaging_focal_length_m"]
            + relay.get("pupil_misconjugation_m", 0.0)
        ),
        nominal_image_distance_m=image_nominal,
    )
    return relay


def flat_dm_opd(config: dict) -> np.ndarray:
    """Reproduce BaldrApp's nominal 0.5-command BMC flat surface."""
    pupil = config["pupil"]
    dm = config["flat_dm"]
    size = int(pupil["array_size"])
    telescope_diameter = float(pupil["physical_diameter_m"])
    padding_factor = size / float(pupil["pixels_across_pupil"])
    # Preserve BaldrApp's historical floor-division coordinate convention.
    lower = (-(telescope_diameter * padding_factor)) // 2
    upper = (telescope_diameter * padding_factor) // 2
    coordinate = np.linspace(lower, upper, size)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)

    actuator_scale = 12 / 10 * (telescope_diameter / 2) / (12 / 2 - 0.5)
    actuator_coordinates = []
    for row in range(12):
        for column in range(12):
            if row in {0, 11} and column in {0, 11}:
                continue
            actuator_coordinates.append((
                (column - 5.5) * actuator_scale + coordinate.mean(),
                (row - 5.5) * actuator_scale + coordinate.mean(),
            ))
    sigma = float(dm["actuator_coupling_factor"]) * actuator_scale
    command_opd = float(dm["flat_command"]) * float(dm["opd_per_command_m"])
    result = np.zeros((size, size), dtype=float)
    for x0, y0 in actuator_coordinates:
        result += command_opd * np.exp(
            -((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / sigma**2
        )
    return result


def relay_intensity(field, wavelength: float, config: dict) -> np.ndarray:
    pupil = config["pupil"]
    relay = derive_relay(config)
    dx = relay["physical_pupil_diameter_m"] / float(pupil["pixels_across_pupil"])
    x, y = _coordinate_grid(field.shape, dx)
    mirror = _angular_spectrum(field, wavelength, dx, relay["distance_to_mirror_m"])
    full_mirror = np.hypot(x, y) <= relay["mirror_diameter_m"] / 2
    angle = relay.get("edge_angle_rad", 0.0)
    rotated_x = x * np.cos(angle) + y * np.sin(angle)
    mirror *= full_mirror & (rotated_x >= relay.get("edge_offset_m", 0.0))
    at_lens = _angular_spectrum(mirror, wavelength, dx, relay["mirror_to_lens_m"])
    after_lens = _thin_lens(at_lens, wavelength, dx, relay["imaging_focal_length_m"])
    at_stop, dx_stop, dy_stop = _one_step_fresnel(
        after_lens, wavelength, dx, relay["imaging_focal_length_m"]
    )
    xs, ys = _coordinate_grid(at_stop.shape, dx_stop, dy_stop)
    stop_radius = np.hypot(
        xs - relay.get("cold_stop_x_offset_m", 0.0),
        ys - relay.get("cold_stop_y_offset_m", 0.0),
    )
    after_stop = at_stop * (stop_radius <= relay["cold_stop_diameter_m"] / 2)
    detector_field, dx_detector, dy_detector = _one_step_fresnel(
        after_stop, wavelength, dx_stop, relay["focus_to_detector_m"], dy_stop
    )
    return np.abs(detector_field) ** 2 * dx_detector * dy_detector / dx**2


def _bin(image: np.ndarray, factor: int) -> np.ndarray:
    if image.shape[0] % factor or image.shape[1] % factor:
        raise ValueError("Image dimensions must be divisible by detector binning.")
    return image.reshape(
        image.shape[0] // factor, factor, image.shape[1] // factor, factor
    ).sum(axis=(1, 3))


def _crop(image: np.ndarray, shape) -> np.ndarray:
    if shape is None:
        return image
    height, width = map(int, shape)
    y0 = (image.shape[0] - height) // 2
    x0 = (image.shape[1] - width) // 2
    if y0 < 0 or x0 < 0:
        raise ValueError("Detector crop is larger than the binned image.")
    return image[y0:y0 + height, x0:x0 + width]


def generate_references(config: dict, material_data_path: Path):
    pupil = make_pupil(config)
    _, profile, _, _ = resolve_source_profile(config)
    amplitude = np.sqrt(float(profile["photons_per_second_per_pixel_per_nm"])) * pupil
    wavelengths, weights_nm = make_spectrum(config)
    dm_opd = flat_dm_opd(config)
    clear_rate = np.zeros_like(pupil)
    masked_rate = np.zeros_like(pupil)
    
    strehl = float(config.get("strehl", 1.0))
    if not np.isfinite(strehl) or not 0.0 <= strehl <= 1.0:
        raise ValueError(f"strehl ratio in the input config file must be finite and lie in [0, 1].")

    for wavelength, weight_nm in zip(wavelengths, weights_nm):
        theta, diameter = mask_parameters(config, float(wavelength), material_data_path)
        clear_rate += weight_nm * relay_intensity(
            zwfs_field(
                amplitude, 0.0, diameter,
                phase=2 * np.pi / float(wavelength) * pupil * dm_opd,
            ),
            float(wavelength), config,
        )
        masked_rate += weight_nm * relay_intensity(
            zwfs_field(
                amplitude, theta, diameter,
                phase=2 * np.pi / float(wavelength) * pupil * dm_opd,
                strehl=strehl,
            ),
            float(wavelength), config,
        )
    detector = config["detector"]
    scale = float(detector["quantum_efficiency"]) * float(detector["exposure_s"])
    clear = _bin(clear_rate, int(detector["binning"])) * scale
    masked = _bin(masked_rate, int(detector["binning"])) * scale
    crop = detector.get("crop")
    return _crop(clear, crop), _crop(masked, crop), wavelengths


def load_config(path: str | Path) -> dict:
    with Path(path).open() as stream:
        return json.load(stream)
