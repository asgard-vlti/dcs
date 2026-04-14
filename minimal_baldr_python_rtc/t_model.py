# %%
import model
import hcipy
import numpy as np

# %%

telescope_diameter = 8.2

wavelength_wfs = 1.6e-6


num_pupil_pixels = 256
pupil_grid_diameter = 2 * telescope_diameter
pupil_grid = hcipy.make_pupil_grid(num_pupil_pixels, pupil_grid_diameter)

lab_diam = 12e-3

detector_grid = hcipy.make_pupil_grid(32, diameter=2 * pupil_grid_diameter)

pupil_grid_lab = hcipy.make_pupil_grid(num_pupil_pixels, 2 * lab_diam)

aperture = hcipy.evaluate_supersampled(
    hcipy.make_vlt_aperture(with_spiders=False), pupil_grid, 6
)
aperture_subsampled = hcipy.subsample_field(aperture, 256 / 32, statistic="sum")
aperture_subsampled /= aperture_subsampled.max()
# amp_err = 0.2 * np.random.randn(*aperture_subsampled.shape)

# tilted amplitude
# amp_err = 1.0 * (detector_grid.x / detector_grid.x.max())
amp_err = 1.0 * (detector_grid.y / detector_grid.y.max())

pupil = hcipy.Field((1 + amp_err) * aperture_subsampled, detector_grid)

hcipy.imshow_field(pupil, detector_grid)
# %%
phasemask_diam = 44e-6
wavels = np.linspace(1.5e-6, 1.7e-6, 5)

ref = model.create_model_reference(phasemask_diam, pupil, wavels)

hcipy.imshow_field(ref, detector_grid)
# %%
