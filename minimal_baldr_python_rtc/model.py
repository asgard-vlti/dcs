import hcipy
import numpy as np

num_pupil_pixels = 256
lab_diam = 12e-3
pupil_grid_lab = hcipy.make_pupil_grid(num_pupil_pixels, 2 * lab_diam)


def pupil_img_to_supersample(pupil_img):
    pupil_img_2d = pupil_img.shaped
    pupil_img_supersampled = np.zeros(pupil_grid_lab.shape)
    for i in range(32):
        for j in range(32):
            pupil_img_supersampled[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = (
                pupil_img_2d[i, j]
            )
    return hcipy.Field(pupil_img_supersampled.ravel(), pupil_grid_lab)


def create_model_reference(phasemask_diam, pupil_img, wavels, phase_shift_deg=90.0):
    ref_wavel = wavels[len(wavels) // 2]

    mag_between_mask_and_stop = 40 / 187
    cold_stop_diameter = 2.1e-3 * mag_between_mask_and_stop

    focal_grid = hcipy.make_focal_grid(
        q=32,
        num_airy=8,
        pupil_diameter=12e-3,
        reference_wavelength=ref_wavel,
        focal_length=254e-3,
    )

    cold_mask = hcipy.make_circular_aperture(cold_stop_diameter)(focal_grid)
    phase_mask = hcipy.make_circular_aperture(phasemask_diam)(focal_grid)

    field_stop_ideal = hcipy.OccultedLyotCoronagraph(
        pupil_grid_lab,
        cold_mask * np.exp(1j * np.radians(phase_shift_deg) * phase_mask),
        focal_plane_mask_grid=focal_grid,
        focal_length=254e-3,
    )

    pupil_img_supersampled = pupil_img_to_supersample(pupil_img)

    img_total = 0.0
    for wavel in wavels:
        wf = hcipy.Wavefront(pupil_img_supersampled, wavel)
        wf.total_power = 1
        wf = field_stop_ideal.forward(wf)

        img = wf.intensity
        img_subsampled = hcipy.subsample_field(img, 256 / 32, statistic="sum")
        img_total += img_subsampled

    return img_total
