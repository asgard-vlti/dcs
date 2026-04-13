import numpy as np
import hcipy
import matplotlib.pyplot as plt

_epsilon = 1e-12


def make_hc_act_grid():
    n_act = 12
    n_beam = 10.0  # the number of actuators the pupil uses

    act_grid = hcipy.make_pupil_grid(n_act, diameter=n_act / n_beam)
    return act_grid


def make_fourier_basis(grid, fourier_grid, sort_by_energy=True):
    """
    Taken from HCIPY, adapted to return the frequencies too

    Make a Fourier basis.

    Fourier modes this function are defined to be real. This means that for each point, both a sine and cosine mode is returned.

    Repeated frequencies will not be repeated in this mode basis. This means that opposite points in the `fourier_grid` will be silently ignored.

    Parameters
    ----------
    grid : Grid
        The :class:`Grid` on which to calculate the modes.
    fourier_grid : Grid
        The grid defining all frequencies.
    sort_by_energy : bool
        Whether to sort by increasing energy or not.

    Returns
    -------
    ModeBasis
        The mode basis containing all Fourier modes.
    """
    modes_cos = []
    modes_sin = []
    energies = []
    ignore_list = []
    freqs = []

    c = np.array(grid.coords)

    for i, p in enumerate(fourier_grid.points):
        if i in ignore_list:
            continue

        mode_cos = hcipy.Field(np.cos(np.dot(p, c)), grid)
        mode_sin = hcipy.Field(np.sin(np.dot(p, c)), grid)

        modes_cos.append(mode_cos)
        modes_sin.append(mode_sin)

        j = fourier_grid.closest_to(-p)

        dist = fourier_grid.points[j] + p
        dist2 = np.dot(dist, dist)

        p_length2 = np.dot(p, p)
        energies.append(p_length2)
        freqs.append(p)

        if dist2 < (_epsilon * p_length2):
            ignore_list.append(j)

    if sort_by_energy:
        ind = np.argsort(energies)
        modes_sin = [modes_sin[i] for i in ind]
        modes_cos = [modes_cos[i] for i in ind]
        freqs = [freqs[i] for i in ind]
        energies = np.array(energies)[ind]

    modes = []
    mode_freqs = []
    for i, E in enumerate(energies):
        # Filter out and correctly normalize zero energy vs non-zero energy modes.
        if E > _epsilon:
            modes.append(modes_cos[i] * np.sqrt(2))
            modes.append(modes_sin[i] * np.sqrt(2))
            mode_freqs.append(freqs[i])
            mode_freqs.append(freqs[i])
        else:
            modes.append(modes_cos[i])
            mode_freqs.append(freqs[i])

    return hcipy.ModeBasis(modes, grid), np.array(mode_freqs)


def pin_outer_edge(basis):
    """
    Pin the outermost row and column of pixels of each basis mode
    to the value of the mode inwards from it

    assumes basis is square
    """
    if isinstance(basis, hcipy.ModeBasis):
        transformation_matrix = basis.transformation_matrix.copy()
    else:
        raise NotImplementedError("Only implemented for ModeBasis for now")

    for i in range(transformation_matrix.shape[1]):
        mode = transformation_matrix[:, i].reshape(basis.grid.shape)
        mode[0, :] = mode[1, :]
        mode[-1, :] = mode[-2, :]
        mode[:, 0] = mode[:, 1]
        mode[:, -1] = mode[:, -2]
        # corners should be zero:
        mode[0, 0] = 0
        mode[0, -1] = 0
        mode[-1, 0] = 0
        mode[-1, -1] = 0
        transformation_matrix[:, i] = mode.flatten()
    return hcipy.ModeBasis(transformation_matrix, basis.grid)


def normalise(basis):
    if isinstance(basis, hcipy.ModeBasis):
        grid = basis.grid
        cur_basis = basis.transformation_matrix.copy()

    norms = np.linalg.norm(cur_basis, axis=0)
    norms[norms < _epsilon] = 1
    cur_basis /= norms[None, :]

    if isinstance(basis, hcipy.ModeBasis):
        return hcipy.ModeBasis(cur_basis, grid)
    return cur_basis


def fourier_basis(
    act_grid,
    min_freq_HO,
    max_freq_HO,
    spacing_HO,
    start_HO,
    orthogonalise=True,
    pin_edges=True,
    scale_by_freq=True,
):
    n_accross = int((max_freq_HO - start_HO) / spacing_HO) + 1

    freqs = []
    u = start_HO
    for i in range(n_accross):
        v = start_HO
        for j in range(n_accross):
            if (
                np.sqrt(u**2 + v**2) <= max_freq_HO
                and np.sqrt(u**2 + v**2) >= min_freq_HO
            ):
                freqs.append([u, v])
                if np.abs(u) > 1e-6 and v > 1e-6:
                    freqs.append([-u, v])
            v += spacing_HO
        u += spacing_HO
    freqs = np.array(freqs)

    HO_modes, freqs_used = make_fourier_basis(act_grid, hcipy.Grid(freqs.T * 2 * np.pi))

    abs_freq_used = np.linalg.norm(freqs_used, axis=1)

    # remove all modes with <=1 freq in total
    keep = abs_freq_used > 1.1 * 2 * np.pi
    keep = np.logical_and(
        keep,
        [
            np.linalg.norm(HO_modes.transformation_matrix[:, n]) > 1e-6
            for n in range(HO_modes.num_modes)
        ],
    )

    HO_modes = hcipy.ModeBasis(HO_modes.transformation_matrix[:, keep], act_grid)
    abs_freq_used = abs_freq_used[keep]
    freqs_used = freqs_used[keep]

    HO_modes = normalise(HO_modes)

    tt_freqs = hcipy.Grid([[0.001, 0], [0, 0.001]])
    tt = hcipy.make_sine_basis(act_grid, tt_freqs)
    tt = normalise(tt)

    combined_transformation_matrix = np.hstack(
        (tt.transformation_matrix, HO_modes.transformation_matrix)
    )
    fourier = hcipy.ModeBasis(combined_transformation_matrix, act_grid)
    freqs_used = np.vstack((tt_freqs.points, freqs_used))
    abs_freq_used = np.linalg.norm(freqs_used, axis=1)

    if orthogonalise:
        fourier = fourier.orthogonalized
        fourier = normalise(fourier)

    if pin_edges:
        fourier = pin_outer_edge(fourier)

    if scale_by_freq:
        max_freq_possible = 5 * 2 * np.pi
        scaling_factor = 1.0 + 1.0 * (abs_freq_used / max_freq_possible) ** 2
        fourier = hcipy.ModeBasis(
            fourier.transformation_matrix * scaling_factor[None, :], act_grid
        )

    return fourier, freqs_used