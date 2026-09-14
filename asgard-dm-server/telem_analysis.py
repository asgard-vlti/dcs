"""
Simple reader and analysis tools for four DM telemetry FITS files.
files are written from asgard-dm-server/asgard_commander_MDM_high_perf_server.c
Benjamin Courtney-Barrer, 14-9-26
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.animation import FuncAnimation
from scipy.signal import welch

import sys
from pathlib import Path

N_TEL, N_ACT, N_SIDE = 4, 140, 12
DM_MASK = np.ones((N_SIDE, N_SIDE), dtype=bool)
DM_MASK[(0, 0, -1, -1), (0, -1, 0, -1)] = False


def square_dm(commands):
    """Put 140-actuator command vectors onto 12 x 12 grids."""
    commands = np.asarray(commands)
    if commands.shape[-1] != N_ACT:
        raise ValueError(f"expected {N_ACT} actuators")
    grids = np.full(commands.shape[:-1] + (N_SIDE, N_SIDE), np.nan)
    grids[..., DM_MASK] = commands
    return grids


class DMTelemetry:
    """Read and analyse four-telescope deformable-mirror telemetry.

    The FITS primary image must contain one row per time sample and 560
    columns: four consecutive 140-actuator DM commands. Archived ``uint16``
    values are converted back to normalized commands with ``value / 65535``.
    The ``DM_RECORD_TIME`` extension must contain the integer columns ``TSEC``
    and ``TNUSEC``.

    Parameters
    ----------
    path : str or pathlib.Path
        Telemetry FITS file.
    basis : {None, "fourier", "zernike", "zonal"}, optional
        Modal basis from the project's :mod:`modal_basis` module. With the
        default ``None``, FITS reading, timing analysis, OPD conversion and
        animation remain available, but modal fitting does not.
    nmodes : int, optional
        Number of modes. Required when ``basis`` is not ``None``.

    Attributes
    ----------
    commands : numpy.ndarray
        Normalized commands with shape ``(samples, 4, 140)``.
    time : numpy.ndarray
        Elapsed time in seconds, starting at zero.
    dt : numpy.ndarray
        Time between consecutive samples in seconds.
    sample_interval, sample_rate : float
        Median positive sample interval and corresponding rate in Hz.
    basis_matrix : numpy.ndarray or None
        Active-actuator basis with shape ``(140, nmodes)``. This is suitable
        for reuse by the supervisor or RTC.

    Examples
    --------
    Read a file using the basic defaults::

        telemetry = DMTelemetry("dms_T00-39-43.fits")

        dm1 = telemetry.dm(1)                # (samples, 140), values in [0, 1]
        dm1_maps = telemetry.dm(1, square=True)  # (samples, 12, 12)
        print(telemetry.sample_rate)

    Find intervals that differ from the median cadence by more than 50%::

        slip_after = telemetry.timing_slips(tolerance=0.5)
        for sample in slip_after:
            print(sample, telemetry.dt[sample])

    Configure a modal basis and fit selected command-space modes::

        telemetry = DMTelemetry(
            "dms_T00-39-43.fits", basis="fourier", nmodes=100
        )
        coefficients = telemetry.coefficients(1)
        telemetry.plot_modes(1, modes=[0, 1, 2])
        telemetry.plot_psd(1, modes=[0, 1, 2], nperseg=1024)

    Restrict the fit to a centred circular pupil. The radius is measured in
    actuator pitches and may be a float::

        coefficients = telemetry.coefficients(
            1, space="command", pupil_radius=4.5
        )

        telemetry.plot_pupil_support(4.5)

    Convert normalized commands to OPD in nanometres. Gaussian influence
    functions have 0.75 nearest-neighbour coupling by default and are
    normalized so a uniform command produces a uniform surface. For a
    reflective DM, ``OPD = 2 * displacement * cos(angle)``::

        opd = telemetry.opd(
            1,
            angle_deg=10,
            opl_per_command_nm=3000,
            coupling=0.75,
        )                               # (samples, 12, 12)

    Increase OPD sampling without changing the physical 12-actuator-wide DM.
    Modal modes are automatically regenerated at the same sampling before an
    OPD-space fit::

        opd = telemetry.opd(1, nsample=48)  # (samples, 48, 48)

        coefficients = telemetry.coefficients(
            1,
            space="opd",
            nsample=48,
            pupil_radius=5.0,
            angle_deg=10,
        )

        telemetry.plot_modes(
            1, [0, 1, 2], space="opd", nsample=48, pupil_radius=5.0
        )
        telemetry.plot_psd(
            1, [0, 1, 2], space="opd", nsample=48, pupil_radius=5.0
        )

    Animate normalized commands, absolute OPD, or OPD changes relative to the
    first frame. Keep the returned animation assigned while it is displayed::

        animation = telemetry.animate(1)

        animation = telemetry.animate(
            1,
            space="opd",
            nsample=48,
            angle_deg=10,
            difference=True,
        )

    Save the same animation as an MP4 (requires FFmpeg) or GIF (requires
    Pillow). Animation options are passed through unchanged::

        telemetry.save_movie(
            1,
            "dm_opd.mp4",
            fps=30,
            dpi=150,
            space="opd",
            nsample=48,
            difference=True,
        )

        telemetry.save_movie(1, "dm_commands.gif", fps=15)

    Trim or subsample the saved frames with Python-style ``start``, ``stop``
    and ``step`` values::

        telemetry.save_movie(
            1, "dm_trimmed.mp4", start=500, stop=1500, step=2, fps=30
        )

    Notes
    -----
    ``difference=False`` shows the absolute DM operating point, including any
    command bias. ``difference=True`` affects animation display only and
    subtracts the first displayed frame. Set ``normalize_influence=False`` in
    :meth:`opd` only when an unnormalized sum of calibrated influence
    functions is specifically required.
    """

    def __init__(self, path, basis=None, nmodes=None):
        self.path = Path(path)
        with fits.open(self.path, memmap=False) as hdus:
            raw = hdus[0].data
            times = hdus["DM_RECORD_TIME"].data
            if raw.ndim != 2 or raw.shape[1] != N_TEL * N_ACT:
                raise ValueError(f"expected primary image shape (samples, 560), got {raw.shape}")
            if len(raw) != len(times):
                raise ValueError("command and timestamp counts differ")
            # Telemetry is archived as round(command * 65535).
            self.commands = raw.reshape(-1, N_TEL, N_ACT).astype(float) / 65535
            self.tsec = times["TSEC"].astype(np.int64)
            self.tnusec = times["TNUSEC"].astype(np.int64)

        self.time_ns = self.tsec * 1_000_000_000 + self.tnusec
        self.time = (self.time_ns - self.time_ns[0]) * 1e-9
        self.dt = np.diff(self.time_ns) * 1e-9
        self.sample_interval = float(np.median(self.dt[self.dt > 0]))
        self.sample_rate = 1 / self.sample_interval

        options = {"fourier": "Fourier", "zernike": "Zernike", "zonal": "Zonal"}
        if basis is not None:
            if not isinstance(basis, str) or basis.lower() not in options:
                raise ValueError("basis must be None, 'fourier', 'zernike' or 'zonal'")
            basis = basis.lower()

        self.basis, self.nmodes, self._basis_generator = basis, nmodes, None
        self.basis_matrix = self.mode_grid = None
        if basis is not None:
            if nmodes is None:
                raise ValueError("nmodes is required when basis is supplied")

            #from baldr_jcr import modal_basis
            #import modal_basis

            # import same basis that was used in controller
            #  we only import this if using analysis tools to minimize risk of breaking for just basic reading in telemetry functions
            dcs_root = Path(__file__).resolve().parents[1]
            baldr_root = dcs_root / "baldr_jcr"

            for path in (dcs_root, baldr_root):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))

            from baldr_jcr import modal_basis
            self._basis_generator = getattr(modal_basis, options[basis])()
            modes = self._basis_generator.modes_on_unit_disk(
                nsamplex=N_SIDE, nmodes=nmodes
            )
            if modes.shape != (N_SIDE**2, nmodes):
                raise ValueError(f"modal_basis returned {modes.shape}, expected (144, {nmodes})")
            self.mode_grid = modes
            self.basis_matrix = modes[DM_MASK.ravel()]

    def dm(self, telescope, square=False):
        """Commands for telescope 1, 2, 3 or 4."""
        if telescope not in range(1, N_TEL + 1):
            raise ValueError("telescope must be 1, 2, 3 or 4")
        commands = self.commands[:, telescope - 1]
        return square_dm(commands) if square else commands

    def timing_slips(self, tolerance=0.5):
        """Indices before intervals differing from the median cadence."""
        return np.flatnonzero(
            (self.dt <= 0)
            | (np.abs(self.dt - self.sample_interval) > tolerance * self.sample_interval)
        )

    def opd(self, telescope, angle_deg=0, opl_per_command_nm=3000,
            coupling=0.75, pitch=1.0, nsample=N_SIDE,
            normalize_influence=True):
        """Convert normalized commands to reflected OPD maps in nanometres.

        Each actuator has a Gaussian influence function. ``coupling`` is its
        response one actuator pitch from the centre. Normalization makes a
        uniform command produce a uniform surface despite Gaussian overlap.
        The reflective-DM factor is ``2*cos(angle_deg)``.
        """
        if not 0 < coupling < 1 or pitch <= 0:
            raise ValueError("coupling must be between 0 and 1 and pitch positive")
        if not 0 <= abs(angle_deg) < 90:
            raise ValueError("absolute angle_deg must be less than 90")
        if not isinstance(nsample, int) or nsample < 2:
            raise ValueError("nsample must be an integer of at least 2")

        actuator_y, actuator_x = np.nonzero(DM_MASK)
        axis = np.linspace(0, N_SIDE - 1, nsample)
        xx, yy = np.meshgrid(axis, axis)
        sigma = pitch / np.sqrt(-np.log(coupling))
        influence = np.exp(
            -((((xx * pitch).ravel()[:, None] - actuator_x * pitch) ** 2
               + ((yy * pitch).ravel()[:, None] - actuator_y * pitch) ** 2) / sigma**2)
        )
        if normalize_influence:
            influence /= influence.sum(axis=1, keepdims=True)
        surface = self.dm(telescope) @ influence.T * opl_per_command_nm
        return (surface * 2 * np.cos(np.deg2rad(angle_deg))).reshape(
            -1, nsample, nsample
        )

    @staticmethod
    def _pupil(radius, nsample=N_SIDE):
        if radius <= 0:
            raise ValueError("pupil_radius must be positive")
        axis = np.linspace(0, N_SIDE - 1, nsample)
        xx, yy = np.meshgrid(axis, axis)
        centre = (N_SIDE - 1) / 2
        return (xx - centre) ** 2 + (yy - centre) ** 2 <= radius**2

    def coefficients(self, telescope, space="command", pupil_radius=None,
                     remove_mean=False, **opd_kwargs):
        """Fit modes in normalized command or OPD space.

        ``pupil_radius`` optionally selects a centred circular support, with
        radius measured in actuator pitches. OPD keyword arguments are passed
        to :meth:`opd` when ``space="opd"``.
        """
        if self.basis_matrix is None:
            raise ValueError("supply basis and nmodes when creating DMTelemetry")
        if space == "command":
            values, modes = self.dm(telescope), self.basis_matrix
            if pupil_radius is not None:
                keep = self._pupil(pupil_radius)[DM_MASK]
                values, modes = values[:, keep], modes[keep]
        elif space == "opd":
            nsample = opd_kwargs.get("nsample", N_SIDE)
            values = self.opd(telescope, **opd_kwargs).reshape(-1, nsample**2)
            modes = self._basis_generator.modes_on_unit_disk(
                nsamplex=nsample, nmodes=self.nmodes
            )
            if modes.shape != (nsample**2, self.nmodes):
                raise ValueError(
                    f"modal_basis returned {modes.shape}, "
                    f"expected ({nsample**2}, {self.nmodes})"
                )
            if pupil_radius is not None:
                keep = self._pupil(pupil_radius, nsample).ravel()
                values, modes = values[:, keep], modes[keep]
        else:
            raise ValueError("space must be 'command' or 'opd'")
        if remove_mean:
            values = values - values.mean(axis=0)
        return np.linalg.lstsq(modes, values.T, rcond=None)[0].T

    def plot_pupil_support(self, pupil_radius, ax=None):
        """Plot the circular command-space pupil over the active actuators."""
        inside = self._pupil(pupil_radius)[DM_MASK]
        actuator_y, actuator_x = np.nonzero(DM_MASK)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.scatter(actuator_x[~inside], actuator_y[~inside], marker="s",
                   s=80, color="0.8", label="excluded")
        ax.scatter(actuator_x[inside], actuator_y[inside], marker="s",
                   s=80, color="C0", label="included")
        ax.add_patch(plt.Circle(((N_SIDE - 1) / 2, (N_SIDE - 1) / 2),
                                pupil_radius, fill=False, color="C1", lw=2))
        ax.set(xlabel="Actuator x", ylabel="Actuator y",
               title=f"Command-space pupil: radius {pupil_radius:g} pitches",
               xlim=(-0.75, N_SIDE - 0.25), ylim=(N_SIDE - 0.25, -0.75),
               aspect="equal")
        ax.set_xticks(range(N_SIDE))
        ax.set_yticks(range(N_SIDE))
        ax.legend()
        return fig, ax

    def plot_modes(self, telescope, modes, **coefficient_kwargs):
        coefficients = self.coefficients(telescope, **coefficient_kwargs)
        fig, ax = plt.subplots()
        for mode in modes:
            ax.plot(self.time, coefficients[:, mode], label=f"mode {mode}")
        ax.set(xlabel="Time [s]", ylabel="Coefficient", title=f"Telescope {telescope}")
        ax.legend()
        return fig, ax

    def plot_psd(self, telescope, modes, nperseg=None, **coefficient_kwargs):
        coefficients = self.coefficients(telescope, **coefficient_kwargs)
        fig, ax = plt.subplots()
        for mode in modes:
            frequency, power = welch(
                coefficients[:, mode], fs=self.sample_rate, nperseg=nperseg
            )
            ax.loglog(frequency[1:], power[1:], label=f"mode {mode}")
        ax.set(xlabel="Frequency [Hz]", ylabel="PSD [coefficient²/Hz]",
               title=f"Telescope {telescope}")
        ax.legend()
        return fig, ax

    def animate(self, telescope, space="command", nsample=N_SIDE,
                interval_ms=None, difference=False, cmap="viridis",
                vmin=None, vmax=None, start=0, stop=None, step=1,
                **opd_kwargs):
        """Animate command or OPD maps.

        ``nsample`` applies to OPD space. Set ``difference=True`` to display
        changes relative to the first frame instead of the absolute shape.
        """
        if space == "command":
            if nsample != N_SIDE:
                raise ValueError("increased nsample is available only in OPD space")
            if opd_kwargs:
                raise ValueError("OPD conversion options require space='opd'")
            grids, unit = self.dm(telescope, square=True), "command"
        elif space == "opd":
            grids = self.opd(telescope, nsample=nsample, **opd_kwargs)
            unit = "OPD [nm]"
        else:
            raise ValueError("space must be 'command' or 'opd'")
        frame_numbers = np.arange(len(grids))[slice(start, stop, step)]
        if not len(frame_numbers):
            raise ValueError("start, stop and step select no animation frames")
        grids = grids[frame_numbers]
        if difference:
            grids = grids - grids[0]

        if vmin is None:
            vmin = np.nanmin(grids)
        if vmax is None:
            vmax = np.nanmax(grids)
        if vmin == vmax:
            vmax = vmin + np.finfo(float).eps

        fig, ax = plt.subplots()
        image = ax.imshow(grids[0], cmap=cmap, vmin=vmin, vmax=vmax)
        first = frame_numbers[0]
        title = ax.set_title(
            f"Telescope {telescope} | frame {first:05d} | t = {self.time[first]:.6f} s"
        )
        fig.colorbar(image, ax=ax, label=unit)
        fig.tight_layout()

        def update(i):
            image.set_array(np.ma.masked_invalid(grids[i]))
            frame = frame_numbers[i]
            title.set_text(
                f"Telescope {telescope} | frame {frame:05d} | t = {self.time[frame]:.6f} s"
            )
            return image, title

        animation = FuncAnimation(
            fig, update, frames=len(grids),
            interval=interval_ms or 1000 * self.sample_interval,
            blit=False,
        )
        # Retaining data also makes the animation straightforward to inspect
        # and prevents temporary arrays from disappearing in notebook backends.
        animation.dm_frames = grids
        animation.dm_frame_numbers = frame_numbers
        return animation

    def save_movie(self, telescope, path, fps=30, dpi=100, **animation_kwargs):
        """Save a command or OPD animation as MP4 or GIF.

        Parameters
        ----------
        telescope : int
            Telescope number from 1 to 4.
        path : str or pathlib.Path
            Output filename ending in ``.mp4`` or ``.gif``.
        fps : float, optional
            Playback frame rate. This changes playback speed, not timestamps.
        dpi : int, optional
            Output resolution in dots per inch.
        **animation_kwargs
            Options accepted by :meth:`animate`, such as ``space``,
            ``nsample``, ``difference``, ``angle_deg``, ``cmap``, ``vmin``
            and ``vmax``. Use ``start``, ``stop`` and ``step`` to select a
            subset of telemetry frames.

        Returns
        -------
        pathlib.Path
            The saved movie path.
        """
        path = Path(path)
        writers = {".mp4": "ffmpeg", ".gif": "pillow"}
        if path.suffix.lower() not in writers:
            raise ValueError("movie path must end in .mp4 or .gif")
        animation = self.animate(telescope, **animation_kwargs)
        animation.save(
            path, writer=writers[path.suffix.lower()], fps=fps, dpi=dpi
        )
        plt.close(animation._fig)
        return path


"""
# e.g. you don't need to give basis, nmodes at input - these only need to be added if you use the analysis tools 
telemetry = DMTelemetry(
    path="/Users/bencb/Downloads/20260913/dms_T00-40-14.fits",
    basis='fourier',
    nmodes=100,
)

t_fl = telemetry.dm(1) # (samples, 140)
t_sq = telemetry.dm(1, square=True) # (samples, 12, 12)
print( t_fl.shape, t_sq.shape)

telemetry.time            # elapsed seconds
telemetry.dt              # timestamp differences
telemetry.sample_rate
telemetry.timing_slips()

telemetry.basis_matrix    # (140, nmodes), reusable in supervisor/RTC
telemetry.coefficients(1) # beam 1 modal coefficients

telemetry.plot_modes(1, [0, 1, 2])

telemetry.plot_modes(1, 
                    modes = [0, 1, 2],             
                    space="command",
                    pupil_radius=5.0)

telemetry.plot_modes(1, 
                    modes = [0, 1, 2],             
                    space="opd",
                    nsample=48,
                    pupil_radius=5.0
                    )

telemetry.plot_psd(
             1,
             modes=[0, 1, 2]) 

telemetry.plot_psd(
             1,
             modes=[0, 1, 2],
             space="command",
             pupil_radius=5.0,
         )

to add sampling to opd so we can account for DM influence function 
use nsample 
telemetry.plot_psd(
             1,
             modes=[0, 1, 2],
             space="opd",
             nsample=48,
             pupil_radius=5.0,
         )

animation = telemetry.animate(1)

animation = telemetry.animate(1)

# more details to convert to opd space
animation = telemetry.animate(
    1,
    space="opd",
    nsample=48,
    angle_deg=10,
    opl_per_command_nm=3000,
    coupling=0.75,
    difference = True 
)

telemetry.animate(1,
        space="opd", 
        nsample=48,
        difference=False, 
        start=500, 
        stop=1500, 
        step=20)

# save as movie
telemetry.save_movie(
    1,
    "dm_opd.mp4",
    fps=30,
    dpi=150,
    space="opd",
    nsample=48,
    difference=True,
    angle_deg=10,
)

fig, ax = telemetry.plot_pupil_support(pupil_radius=4.5)
"""

