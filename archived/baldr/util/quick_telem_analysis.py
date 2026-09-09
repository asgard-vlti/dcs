
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import toml
import toml
from mpl_toolkits.axes_grid1 import make_axes_locatable

# PARENT = Path("/home/benjamin/baldr_comissioning").resolve()
# if str(PARENT) not in sys.path:
#     sys.path.insert(0, str(PARENT))
# from utils import utilities as util

import utilities as util




# set_telem_capacity [10000]

# /home/asg/Music/telemetry_20250912_011231.fits


# set_rtc_field "inj_signal.basis_index",0

#  /home/asg/Music/telemetry_20250912_011558.fits

# set_rtc_field "inj_signal.basis_index",2

#  /home/asg/Music/telemetry_20250912_011648.fits

#/home/asg/Music/telemetry_20250912_012533.fits
# open the fits

# New telemetry index compiled from the attached log.
# Assumptions:
# - mask="H3" unless noted; beam=2 for all entries.
# - For CL runs, only TT ki is set per the log; TT kp,kd=0.0; HO all zeros unless stated.
# - Left file extensions exactly as in the log (one is ".fit" without "s").



def load_config(beam_list, folder, phasemask, strict=False):
    """
    Load per-beam configuration TOMLs and return a dict keyed by beam id.

    Parameters
    ----------
    beam_list : Iterable[int|str]
        Beam identifiers (e.g., [0,1,2,3,4] or ["0","1",...]).
    folder : str or Path
        Directory containing baldr_config_<beam>.toml files.
    phasemask : str
        Name of the phasemask block to read under each beam (e.g., "zwfs").
    strict : bool, default False
        If True, raise on missing expected fields; otherwise store None.

    Returns
    -------
    baldr_dict : dict
        {
          <beam_id>: {
            "baldr_pupils": ...,
            "I2A": np.ndarray or None,
            "pupil_masks": {"mask": ..., "exterior": ..., "secondary": ...},
            "ctrl_model": {
                "IM": ..., "I2M": ..., "I2M_LO": ..., "I2M_HO": ...,
                "M2C": ..., "M2C_LO": ..., "M2C_HO": ...,
                "I0": ..., "N0": ..., "N0i": ...,
                "inner_pupil_filt": ...,
                "camera_config": dict or None,
                "bad_pixel_mask": ...,
                "DM_flat": ...
            },
            "strehl_model": {"secondary": ..., "exterior": ...},
            "toml_path": "<full path>"
          },
          ...
        }
    """

    def to_array(val, dtype=None):
        if val is None:
            return None
        arr = np.array(val)
        return arr.astype(dtype) if dtype is not None else arr

    def deep_get(d, keys, default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                if strict:
                    path_str = " → ".join(map(str, keys))
                    raise KeyError(f"Missing key path: {path_str}")
                return default
            cur = cur[k]
        return cur

    folder = Path(folder)
    baldr_dict = {}

    for beam in beam_list:
        beam_id = str(beam).strip()
        toml_path = folder / f"baldr_config_{beam_id}.toml"
        if not toml_path.exists():
            if strict:
                raise FileNotFoundError(f"Config not found: {toml_path}")
            # skip missing beam file
            continue

        with open(toml_path, "r") as f:
            config = toml.load(f)

        # Global-level (not per-beam)
        baldr_pupils = deep_get(config, ["baldr_pupils"], default=None)

        # Per-beam root
        beam_key = f"beam{beam_id}"

        I2A = to_array(deep_get(config, [beam_key, "I2A"], default=None), dtype=float)

        # Pupil masks
        pupil_mask = to_array(deep_get(config, [beam_key, "pupil_mask", "mask"], default=None), dtype=bool)
        exter_mask = to_array(deep_get(config, [beam_key, "pupil_mask", "exterior"], default=None), dtype=bool)
        secon_mask = to_array(deep_get(config, [beam_key, "pupil_mask", "secondary"], default=None), dtype=bool)

        # Control model block under the chosen phasemask
        base = [beam_key, phasemask, "ctrl_model"]

        IM       = to_array(deep_get(config, base + ["IM"],       default=None), dtype=float)
        I2M_raw  = to_array(deep_get(config, base + ["I2M"],      default=None), dtype=float)
        I2M_LO   = to_array(deep_get(config, base + ["I2M_LO"],   default=None), dtype=float)
        I2M_HO   = to_array(deep_get(config, base + ["I2M_HO"],   default=None), dtype=float)
        M2C      = to_array(deep_get(config, base + ["M2C"],      default=None), dtype=float)
        M2C_LO   = to_array(deep_get(config, base + ["M2C_LO"],   default=None), dtype=float)
        M2C_HO   = to_array(deep_get(config, base + ["M2C_HO"],   default=None), dtype=float)
        I0       = to_array(deep_get(config, base + ["I0"],       default=None), dtype=float)
        N0       = to_array(deep_get(config, base + ["N0"],       default=None), dtype=float)
        N0i      = to_array(deep_get(config, base + ["norm_pupil"], default=None), dtype=float)

        inner_pupil_filt = to_array(deep_get(config, base + ["inner_pupil_filt"], default=None))  # leave dtype flexible

        camera_config  = deep_get(config, base + ["camera_config"], default=None)  # keep dict
        bad_pixel_mask = to_array(deep_get(config, base + ["bad_pixel_mask"], default=None))
        dm_flat        = to_array(deep_get(config, base + ["DM_flat"], default=None))

        # Strehl model (per-beam, not necessarily under phasemask)
        strehl_sec = to_array(deep_get(config, [beam_key, "strehl_model", "secondary"], default=None), dtype=float)
        strehl_ext = to_array(deep_get(config, [beam_key, "strehl_model", "exterior"],  default=None), dtype=float)

        baldr_dict[beam_id] = {
            "toml_path": str(toml_path),
            "baldr_pupils": baldr_pupils,
            "I2A": I2A,
            "pupil_masks": {
                "mask": pupil_mask,
                "exterior": exter_mask,
                "secondary": secon_mask,
            },
            "ctrl_model": {
                "IM": IM,
                "I2M": I2M_raw,
                "I2M_LO": I2M_LO,
                "I2M_HO": I2M_HO,
                "M2C": M2C,
                "M2C_LO": M2C_LO,
                "M2C_HO": M2C_HO,
                "I0": I0,
                "N0": N0,
                "N0i": N0i,
                "inner_pupil_filt": inner_pupil_filt,
                "camera_config": camera_config,
                "bad_pixel_mask": bad_pixel_mask,
                "DM_flat": dm_flat,
            },
            "strehl_model": {
                "secondary": strehl_sec,
                "exterior": strehl_ext,
            },
        }

    return baldr_dict





### CLuster analysis on power law fits to classify the pupil regimes and try exract a meaninful filter

import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---- optional sklearn (preferred) -------------------------------------------
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False
    warnings.warn("scikit-learn not found; using a simple NumPy k-means fallback.")

# ---- DM helpers (12x12 grid with missing corners) ---------------------------
def _dm_idx_map_12x12():
    n = 12
    idx_map = -np.ones((n, n), dtype=int)
    k = 0
    for i in range(n):
        for j in range(n):
            if (i, j) in [(0,0), (0,n-1), (n-1,0), (n-1,n-1)]:
                continue
            idx_map[i, j] = k
            k += 1
    if k != 140:
        raise RuntimeError("Index map error.")
    return idx_map

def _labels_grid_from_140(labels_140, flip_ud=False):
    """Map length-140 labels to 12x12 grid (corners NaN)."""
    labels_140 = np.asarray(labels_140).reshape(-1)
    if labels_140.size != 140:
        raise ValueError("labels_140 must have length 140.")
    idx_map = _dm_idx_map_12x12()
    grid = np.full(idx_map.shape, np.nan)
    m = idx_map >= 0
    grid[m] = labels_140[idx_map[m]]
    if flip_ud:
        grid = np.flipud(grid)
    return grid

# ---- scaling utilities -------------------------------------------------------
def _robust_scale(X):
    """Median/IQR scaling; falls back to std if IQR=0."""
    med = np.median(X, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    # avoid divide-by-zero
    alt = np.std(X, axis=0, ddof=1)
    iqr = np.where(iqr <= 0, np.where(alt > 0, alt, 1.0), iqr)
    Xs = (X - med) / iqr
    return Xs, (med, iqr)

def _inv_robust_scale(Z, params):
    med, iqr = params
    iqr = np.where(iqr == 0, 1.0, iqr)
    return Z * iqr + med

# ---- minimal NumPy k-means fallback -----------------------------------------
def _kmeans_np(X, n_clusters, n_init=10, max_iter=300, rng=0):
    rng = np.random.default_rng(rng)
    best_inertia = np.inf
    best = None
    for _ in range(n_init):
        # init centers by random rows
        centers = X[rng.choice(X.shape[0], n_clusters, replace=False)]
        for _it in range(max_iter):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)
            new_centers = np.vstack([X[labels == k].mean(axis=0) for k in range(n_clusters)])
            if np.allclose(new_centers, centers, atol=1e-6, rtol=0):
                centers = new_centers
                break
            centers = new_centers
        inertia = np.sum((X - centers[labels])**2)
        if inertia < best_inertia:
            best_inertia = inertia
            best = (labels, centers)
    labels, centers = best
    return labels, centers, best_inertia

# ---- main clustering (single dataset) ----------------------------------------
def cluster_psd_params_single(alpha, B, r2, *,
                              n_clusters=3,
                              use_logB=True,
                              r2_min=None,
                              scaler="robust",
                              random_state=0,
                              plot_dm_map=True,
                              title=None,
                              savefig=None):
    """
    Cluster actuators using features [alpha, log10(B) (or B), r2].

    Parameters
    ----------
    alpha, B, r2 : (N,) arrays
        Fitted parameters per actuator. B must be > 0 if use_logB=True.
    n_clusters : int
        Number of clusters (>=2 recommended).
    use_logB : bool
        Use log10(B) as a feature. If False, uses raw B (be careful with scaling).
    r2_min : float or None
        If set, exclude points with r2 < r2_min from clustering (labels=-1).
    scaler : {"robust","none"}
        Feature scaling method. "robust" = median/IQR (recommended).
    random_state : int
        Seed for reproducibility (sklearn and fallback).
    plot_dm_map : bool
        If True and N==140, show a 12x12 label map.
    title : str or None
        Title for the DM map plot.
    savefig : str or Path or None
        If provided, save the DM map image.

    Returns
    -------
    out : dict
        {
          "labels": (N,) array of int (-1 for excluded),
          "centers_scaled": (K, 3) in scaled feature space,
          "centers_original": (K, 3) back in [alpha, log10B/B, r2] space,
          "features_scaled": (M, 3) used for fit (M=#valid),
          "features_original": (M, 3),
          "valid_mask": (N,) boolean,
          "silhouette": float or np.nan,
          "model": (KMeans object) or None,
        }
    """
    alpha = np.asarray(alpha, float)
    B = np.asarray(B, float)
    r2 = np.asarray(r2, float)
    N = alpha.size
    if not (B.size == N and r2.size == N):
        raise ValueError("alpha, B, r2 must have the same length.")

    # validity mask
    m = np.isfinite(alpha) & np.isfinite(B) & np.isfinite(r2)
    if use_logB:
        m &= (B > 0)
    if r2_min is not None:
        m &= (r2 >= float(r2_min))

    # build features
    f2 = np.log10(B) if use_logB else B
    X = np.column_stack([alpha, f2, r2])
    Xv = X[m]  # valid
    if Xv.shape[0] < max(2, n_clusters):
        raise ValueError("Not enough valid actuators for clustering.")

    # scale
    if scaler == "robust":
        Xs, scal_params = _robust_scale(Xv)
    elif scaler == "none":
        Xs, scal_params = Xv.copy(), (np.zeros(3), np.ones(3))
    else:
        raise ValueError("scaler must be 'robust' or 'none'.")

    # k-means
    if _HAVE_SK:
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        labels_v = km.fit_predict(Xs)
        centers_s = km.cluster_centers_
        model = km
        sil = silhouette_score(Xs, labels_v) if n_clusters > 1 else np.nan
    else:
        labels_v, centers_s, _ = _kmeans_np(Xs, n_clusters, n_init=10, rng=random_state)
        model = None
        sil = np.nan

    # assemble full labels vector (invalid/excluded -> -1)
    labels = np.full(N, -1, dtype=int)
    labels[m] = labels_v

    # centers back to original feature space
    centers_o = _inv_robust_scale(centers_s, scal_params)

    out = dict(
        labels=labels,
        centers_scaled=centers_s,
        centers_original=centers_o,
        features_scaled=Xs,
        features_original=Xv,
        valid_mask=m,
        silhouette=float(sil),
        model=model,
    )

    # optional DM map (only makes sense for 140-actuator case)
    if plot_dm_map and N == 140:
        grid = _labels_grid_from_140(labels, flip_ud=False)
        k = int(np.nanmax(grid)) + 1 if np.isfinite(grid).any() else n_clusters
        cmap = ListedColormap(plt.cm.tab10.colors[:max(k, 1)])
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=k - 0.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.xaxis.tick_top()
        ax.set_title(title or f"DM clusters (K={n_clusters}, sil={out['silhouette']:.2f})")
        # colorbar with integer ticks
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
        cbar.set_ticks(np.arange(0, k, 1)); cbar.set_ticklabels([str(i) for i in range(k)])
        if savefig:
            fig.savefig(savefig, dpi=300, bbox_inches="tight")
        plt.show()



    return out

# ---- convenience wrapper for your res_batch dict -----------------------------
def cluster_from_res_batch(res_batch, *,
                           n_clusters=3,
                           use_logB=True,
                           r2_min=None,
                           scaler="robust",
                           random_state=0,
                           plot_each=True):
    """
    Run clustering for each dataset in res_batch from fit_psd_powerlaw_batch(..., return_arrays=True).

    Parameters
    ----------
    res_batch : dict
        { dataset_key: (results_list, arrays_dict), ... }
        arrays_dict must contain 'alpha', 'B', 'r2'.
    plot_each : bool
        If True, produce a DM map per dataset (when length==140).

    Returns
    -------
    clustered : dict
        { dataset_key: output_dict_from_cluster_psd_params_single, ... }
    """
    clustered = {}
    for key, (_results, arrays) in res_batch.items():
        alpha = arrays["alpha"]
        B = arrays["B"]
        r2 = arrays["r2"]
        out = cluster_psd_params_single(
            alpha, B, r2,
            n_clusters=n_clusters,
            use_logB=use_logB,
            r2_min=r2_min,
            scaler=scaler,
            random_state=random_state,
            plot_dm_map=plot_each,
            title=f"{key} clusters",
        )
        clustered[key] = out
    return clustered




def spatial_intensity_stat( d , label="clear pupil", **kwargs):
        
    vlims = kwargs.get('vlims', [[0,3000], [0,3000]])
    ylabels = kwargs.get('ylabels', ['y [px]', 'y [act.]'])
    xlabels = kwargs.get('xlabels', ['x [px]', 'x [act.]'])
    cbar_labels = kwargs.get('cbar_labels', ['intensity [adu]', 'intensity [adu]'])
    fontsize = kwargs.get('fontsize', 15)
    cbar_orientation = kwargs.get('cbar_orientation', 'bottom')
    axis_off = kwargs.get('axis_off', True)
    savefig = kwargs.get('savefig', None)


    img_list_0 = [ d[1].data['IMG'][0].reshape(32,32), 
                util.get_DM_command_in_2D( d[1].data['IMG_DM'][0]) ] 
                

    img_list_mean = [ np.mean( d[1].data['IMG'].reshape(-1,32,32),axis=0), 
                util.get_DM_command_in_2D( np.mean( d[1].data['IMG_DM'],axis=0)) ] 
                

    img_list_std = [ np.std( d[1].data['IMG'].reshape(-1,32,32),axis=0), 
                util.get_DM_command_in_2D( np.std( d[1].data['IMG_DM'],axis=0)) ] 
                

    titles = [f'{label} pupil sample', 'projected on DM space']
    util.nice_heatmap_subplots( img_list_0 ,
                        xlabel_list=xlabels, 
                        ylabel_list=ylabels, 
                        title_list= titles, 
                        cbar_label_list=cbar_labels, 
                        fontsize=15, 
                        cbar_orientation = cbar_orientation, 
                        axis_off=axis_off, 
                        vlims=vlims , 
                        savefig=savefig)


    titles = [f'{label} pupil mean', 'projected on DM space']
    util.nice_heatmap_subplots( img_list_mean ,
                        xlabel_list=xlabels, 
                        ylabel_list=ylabels, 
                        title_list= titles, 
                        cbar_label_list=cbar_labels, 
                        fontsize=15, 
                        cbar_orientation = 'bottom', 
                        axis_off=True, 
                        vlims=vlims , 
                        savefig=None)


    titles = [f'{label} std', 'projected on DM space']
    vlims = [[0,1000], [0,1000]]
    util.nice_heatmap_subplots( img_list_std ,
                        xlabel_list=xlabels, 
                        ylabel_list=ylabels, 
                        title_list= titles, 
                        cbar_label_list=cbar_labels, 
                        fontsize=15, 
                        cbar_orientation = 'bottom', 
                        axis_off=True, 
                        vlims=vlims , 
                        savefig=None)# Scinillation analysis


    # active pupil filter 
    dm_pup_filt = util.get_circle_DM_command(radius=4).astype(bool).reshape(-1)

    print(f"spatial median intensity variation in the {label} pupil is \n ", np.median(np.std( d[1].data['IMG_DM'], axis=0)[dm_pup_filt] )/np.median( np.mean( d[1].data['IMG_DM'], axis=0)[dm_pup_filt] )  )

    print(f"spatial max intensity variation in the {label}  pupil is \n ", np.max(np.std( d[1].data['IMG_DM'], axis=0)[dm_pup_filt] )/np.min( np.mean( d[1].data['IMG_DM'], axis=0)[dm_pup_filt] )  )

    #scint = np.std( d[1].data['IMG_DM'], axis=0)[dm_pup_filt] / np.mean( d_clear[1].data['IMG_DM'], axis=0)[dm_pup_filt] 


# robust bad pixel removal

try:
    from scipy.ndimage import median_filter, binary_dilation
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

def _mad(a, axis=None):
    """Median Absolute Deviation (scaled to σ for Gaussian with 1.4826)."""
    med = np.nanmedian(a, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(a - med), axis=axis, keepdims=True)
    return med, mad * 1.4826  # robust sigma

def _robust_z(x, med, sig, eps=1e-12):
    return (x - med) / (sig + eps)

def _spatial_median(frame, ksize=3):
    if _HAVE_SCIPY:
        return median_filter(frame, size=ksize, mode="mirror")
    # NumPy fallback (3x3) using sliding windows (fast enough for moderate sizes)
    if ksize != 3:
        raise ValueError("ksize!=3 requires SciPy; install scipy for general sizes.")
    # pad
    pad = 1
    fp = np.pad(frame, pad_width=pad, mode="edge")
    # build neighborhood stack
    neigh = []
    for di in (-1,0,1):
        for dj in (-1,0,1):
            neigh.append(fp[pad+di:pad+di+frame.shape[0],
                             pad+dj:pad+dj+frame.shape[1]])
    neigh = np.stack(neigh, axis=0)
    return np.nanmedian(neigh, axis=0)

def detect_and_fix_bad_pixels(
    cube: np.ndarray,
    temporal_z_thresh: float = 6.0,   # robust |z| on per-pixel time series (hot/dead/noisy)
    temporal_low_var_frac: float = 0.02,  # bottom X% variance pixels (stuck)
    spatial_z_thresh: float = 6.0,    # per-frame robust |z| vs local spatial background
    spatial_kernel: int = 5,          # spatial median kernel for background
    grow_radius: int = 1,             # grow bad mask by ~this many pixels
    replace_ksize: int = 3            # spatial median window used for replacement
):
    """
    Inputs
    ------
    cube : (T,H,W) array
    temporal_z_thresh : robust z threshold for temporal outliers (global over time per pixel)
    temporal_low_var_frac : flag bottom X% (0..1) of robust sigma over time as stuck
    spatial_z_thresh : robust z threshold for per-frame spatial outliers
    spatial_kernel : size (odd) of spatial background kernel (median)
    grow_radius : rounds of 4-neighbor dilation (1 ~ one pixel) to expand mask
    replace_ksize : size (odd) of spatial median for replacement

    Returns
    -------
    bad_mask : (H,W) boolean mask (True=bad pixel)
    cleaned : (T,H,W) array with bad pixels repaired
    stats : dict with summary metrics
    """
    assert cube.ndim == 3, "cube must be (T,H,W)"
    T, H, W = cube.shape
    cube = cube.astype(np.float64, copy=False)

    # ---------- 1) Temporal robust stats per pixel ----------
    # med_t, sig_t have shape (1,H,W)
    med_t, sig_t = _mad(cube, axis=0)
    # robust z across time: |z| > thresh at *any* time and/or persistent anomalies
    # We also compute a robust variance proxy = (sig_t.squeeze()).
    sig_map = np.squeeze(sig_t, axis=0)  # (H,W)
    # extremely low variance (stuck) → mark bottom X%
    finite_sig = sig_map[np.isfinite(sig_map)]
    cutoff = np.nanpercentile(finite_sig, temporal_low_var_frac * 100) if finite_sig.size else 0.0
    stuck_mask = sig_map <= cutoff

    # Pixels that repeatedly go extreme: check fraction of frames exceeding temporal_z_thresh
    z_t = _robust_z(cube, med_t, sig_t)  # (T,H,W)
    frac_extreme = np.mean(np.abs(z_t) > temporal_z_thresh, axis=0)  # (H,W)
    temporal_mask = (frac_extreme > 0.2) | stuck_mask  # >20% extreme frames or stuck

    # ---------- 2) Spatial robust stats per frame ----------
    spatial_mask = np.zeros((H, W), dtype=bool)
    # Use a coarse spatial median background per frame; flag big residuals
    k = spatial_kernel if spatial_kernel % 2 else spatial_kernel + 1
    for t in range(T):
        frame = cube[t]
        # Build spatial background
        bg = _spatial_median(frame, ksize=k)
        res = frame - bg
        med_s, sig_s = _mad(res, axis=None)  # scalars for the whole frame (robust)
        z_s = _robust_z(res, med_s, sig_s)
        # per-frame candidate bads
        bad_t = np.abs(z_s) > spatial_z_thresh
        # accumulate pixels that are repeatedly spatial outliers across time
        spatial_mask |= bad_t

    # Keep only pixels that are spatial outliers somewhat persistently
    # (avoid flagging a single random hit); require > 5% of frames
    # Note: recompute persistence properly
    spatial_persist = np.zeros((H, W), dtype=np.float64)
    for t in range(T):
        frame = cube[t]
        bg = _spatial_median(frame, ksize=k)
        res = frame - bg
        med_s, sig_s = _mad(res, axis=None)
        z_s = _robust_z(res, med_s, sig_s)
        spatial_persist += (np.abs(z_s) > spatial_z_thresh)
    spatial_mask = (spatial_persist / T) > 0.05

    # ---------- 3) Combine with NaNs and grow mask ----------
    bad_mask = temporal_mask | spatial_mask | ~np.isfinite(np.nanmedian(cube, axis=0))

    if grow_radius > 0:
        if _HAVE_SCIPY:
            # 4-neighbor dilation repeated grow_radius times
            struct = np.array([[0,1,0],
                               [1,1,1],
                               [0,1,0]], dtype=bool)
            for _ in range(grow_radius):
                bad_mask = binary_dilation(bad_mask, structure=struct)
        else:
            # Simple NumPy dilation (4-neighbor) grow_radius times
            for _ in range(grow_radius):
                up    = np.pad(bad_mask[1:,:], ((0,1),(0,0)), mode="edge")
                down  = np.pad(bad_mask[:-1,:], ((1,0),(0,0)), mode="edge")
                left  = np.pad(bad_mask[:,1:], ((0,0),(0,1)), mode="edge")
                right = np.pad(bad_mask[:,:-1], ((0,0),(1,0)), mode="edge")
                bad_mask = bad_mask | up | down | left | right

    # ---------- 4) Repair: spatiotemporal median ----------
    cleaned = cube.copy()
    # Precompute temporal median for fallback
    tmed = np.nanmedian(cube, axis=0)  # (H,W)

    # Spatial replacement per frame at bad pixels
    rep_k = replace_ksize if replace_ksize % 2 else replace_ksize + 1
    for t in range(T):
        frame = cleaned[t]
        # spatial candidate
        spatial_med = _spatial_median(frame, ksize=rep_k)
        # where mask is bad, replace with spatial median; if that equals original (rare) or NaN, fall back to temporal median
        repl = spatial_med
        # fallback where repl is NaN or unchanged (edge pathological cases)
        need_fallback = ~np.isfinite(repl) | (repl == frame)
        # Apply replacements
        f = frame.copy()
        f[bad_mask] = repl[bad_mask]
        # fallback where necessary
        fb = need_fallback & bad_mask
        if np.any(fb):
            f[fb] = tmed[fb]
        cleaned[t] = f

    stats = {
        "T": T, "H": H, "W": W,
        "bad_fraction": float(np.mean(bad_mask)),
        "temporal_bad_fraction": float(np.mean(temporal_mask)),
        "spatial_bad_fraction": float(np.mean(spatial_mask)),
        "stuck_fraction": float(np.mean(stuck_mask)),
        "temporal_z_thresh": temporal_z_thresh,
        "spatial_z_thresh": spatial_z_thresh,
        "spatial_kernel": k,
        "grow_radius": grow_radius,
        "replace_ksize": rep_k,
    }
    return bad_mask, cleaned, stats



def get_data(data_key, return_rtc_config=False):

    d = fits.open(data_dict[data_key]["data"])
    
    beam = data_dict[data_key]["beam"]
    mask = data_dict[data_key]["mask"]

    if return_rtc_config:
        rtc_configs = load_config(beam_list=[beam], folder="/home/benjamin/baldr_comissioning/14-9-25night5/", phasemask=mask, strict=False)
        return d, rtc_configs
    else:
        return d
    

def order_labels_by_radius(labels2d):
    Z = np.asarray(labels2d)
    H, W = Z.shape
    yy, xx = np.indices((H, W))
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r = np.hypot(yy - cy, xx - cx)

    # compute mean radius per cluster id (>=0)
    ids = np.array(sorted({int(i) for i in np.unique(Z) if i >= 0}))
    means = np.array([r[Z == k].mean() if np.any(Z == k) else np.inf for k in ids])

    order = ids[np.argsort(means)]                 # old ids ordered by mean radius
    remap = {old: new for new, old in enumerate(order)}

    Z_new = Z.copy()
    for old, new in remap.items():
        Z_new[Z == old] = new                      # reindex clusters 0..K-1 by radius
    # keep -1 as is
    return Z_new, order, means[np.argsort(means)]



###############################################

# put file1 and file2 as argparser arguments inputs 
# example usage: python script.py --file1 --file2

# Idea is we compare some statistics between two datasets

import argparse
from astropy.io import fits

parser = argparse.ArgumentParser(description="Compare two TOML files and summarize base items.")
parser.add_argument("file1", type=Path, help="First fits telemetry file from baldr (closed)")
parser.add_argument("file2", type=Path, help="fits telemetry file frome baldr (open)" )

args = parser.parse_args()

file1 = args.file1
file2 = args.file2

data_dict = {
    "open" : {"data": f"{file1}", "beam": 1, "mask": "H3"},
    "close" : {"data": f"{file2}", "beam": 2, "mask": "H3"},
}



res_batch = {}
for k in ['close','open']: #["OL_b2_020212"] : #"CL_TTki0p05_b2_020636"]:
    d = get_data(data_key = k, return_rtc_config=False)
            
    t_list = []
    y_list = []
    for idx in range(d[1].data["IMG"].shape[1]):       
        t = 1e-6 * (d[1].data["TIMESTAMPS"]  - d[1].data["TIMESTAMPS"][0] )
        y = d[1].data["IMG"][:,idx]

        #f, Pxx = util.psd_welch_single(t, y, average="median",fix_nonuniform='resample_linear')
        t_list.append( t )
        y_list.append( y )

    f_list, psd_list = util.psd_welch_batch(t_list, y_list, average="median",fix_nonuniform='resample_linear' )

    # # check a single fit 
    # _  = util.fit_psd_powerlaw(f_list[65],
    #                             psd_list[65], 
    #                             fmin=20, 
    #                             fmax=500, 
    #                             plot=True, 
    #                             color="C3",
    #                             max_fraction_den=20, show_B=True)


    res_batch[k] = util.fit_psd_powerlaw_batch(
        f_list, psd_list, 
        fmin=20, fmax=500,
        log_base=10.0,
        return_arrays=True
    )



# After you build `res_batch` as you showed:
clustered = cluster_from_res_batch(
    res_batch,
    n_clusters=5,
    use_logB=True,      # cluster on [alpha, log10(B), r2]
    r2_min=None,        # or e.g. 0.6 to ignore poor fits
    scaler="robust",
    random_state=0,
    plot_each=True      # shows a DM 12x12 map for each dataset
)

# # Access labels for a dataset:
# labels = clustered["zwfs_pup_OL_b2_1"]["labels"]        # length 140, -1 for excluded
# centers = clustered["zwfs_pup_OL_b2_1"]["centers_original"]  # shape (K, 3): [alpha, log10B (or B), r2]

# # If you want to visualize centers in physical units:
# centers_alpha = centers[:, 0]
# centers_logB  = centers[:, 1]    # if use_logB=True; otherwise this is B
# centers_r2    = centers[:, 2]



for k in res_batch:
    # plt.figure()
    # plt.title(k)
    # plt.imshow( clustered[k]["labels"].reshape(32,32) )
    # plt.colorbar()

    from matplotlib.colors import ListedColormap, BoundaryNorm

    labels2d = clustered[k]["labels"].reshape(32,32) 
    Z_ord, order, mean_r = order_labels_by_radius(labels2d)

    res_batch[k][1]["cluster_order"] = order
    res_batch[k][1]["cluster_mean_radius"] = mean_r
    res_batch[k][1]["labels_ordered"] = Z_ord.reshape(-1)
    
    #Z = labels2d  # integers 0..K-1 (shape HxW)
    K = int(Z_ord.max()) + 1
    cmap = ListedColormap(plt.get_cmap("tab20").colors[:K])
    norm = BoundaryNorm(np.arange(-0.5, K+0.5, 1), K)

    # im = plt.imshow(Z_ord, cmap=cmap, norm=norm, interpolation="nearest")
    # cbar = plt.colorbar(im, ticks=np.arange(0, K)); cbar.ax.set_yticklabels([str(i) for i in range(K)])
    # plt.show()


strehl_pixel_cluster = 2
dark_pixel_cluster = 4
active_pupil_cluster = 0
strehl_signal = {} # to hold the final estimate for comparison
fig,ax = plt.subplots(2,sharex=False)
for ii, k in enumerate(res_batch):
    exterior_mask = (res_batch[k][1]["labels_ordered"]== strehl_pixel_cluster).reshape(32,32)

    
    d = get_data(data_key = k, return_rtc_config=False)

    raw_imgs = d[1].data["IMG"]

    dark = np.mean(  raw_imgs[ : , res_batch[k][1]["labels_ordered"]== dark_pixel_cluster  ] )

    mean_pupil = np.mean( raw_imgs[ : ,  res_batch[k][1]["labels_ordered"]== active_pupil_cluster ] - dark )
    
    imgs = ( raw_imgs - dark ) / mean_pupil 

    # bad pixel detection and filtering 
    # bad_mask, imgs, stats = detect_and_fix_bad_pixels( raw_imgs.reshape(-1,32,32),
    #                           temporal_z_thresh=1.0,
    #                           temporal_low_var_frac=0.02,
    #                           spatial_z_thresh=1.0, 
    #                             spatial_kernel=1,
    #                             grow_radius=1,
    #                             replace_ksize=1)

    ext_signal = []
    for i in range(imgs.shape[0]):
        ext_signal.append( np.mean( imgs[i].reshape(32,32)[exterior_mask] )   ) # normalize by mean pupil intensity to get relative signal

    e_TT_m = np.mean( d[1].data["e_LO"],axis=1 )

    ax[0].hist(ext_signal, bins=50, alpha=0.5, label=f"{k}, std={np.std(ext_signal):.4f},avg={np.mean(ext_signal):.4f}")
    ax[1].hist( e_TT_m, bins=np.linspace(np.min(e_TT_m),np.max(e_TT_m),50), alpha=0.5, label=f"{k}, std={np.std(e_TT_m):.4f},avg={np.mean(e_TT_m):.4f}")


    ax[0].set_xlabel("mean exterior signal [adu]")
    ax[1].set_xlabel("e_TT [unitless]")
    ax[0].set_ylabel("counts")

for axx in ax.reshape(-1):
    axx.set_ylabel("counts")
    axx.legend()

plt.show()

    # lucky_cutoff = np.quantile( ext_signal , 0.98) #10k samples => 99th perc. keeps 100 samples 

    # unlucky_cutoff = np.quantile( ext_signal , 0.20)


    # lucky_ext_signals = np.array(ext_signal)[ext_signal > lucky_cutoff]

    # # if np.mean(lucky_ext_signals) < 0.1 * np.mean( I0[exterior_mask] ) : # if the lucky exterior signals are less than 10% of the internal reference exterior signal then we warn the user 
    # #     print("WARNING : lucky exterior signals are less than 10% of the internal reference exterior signal. May not be a good reference")

    # lucky_imgs = np.array(imgs)[ext_signal > lucky_cutoff]

    # unlucky_imgs = np.array(imgs)[ext_signal < unlucky_cutoff]


    # # geet new zwfs reference (I0) intensity from luck images onsky 

    # I0_bad = np.mean( unlucky_imgs, axis=0)
    # I0_unlucky_ref = I0_bad / np.sum( I0_bad )

    # I0_new = np.mean( lucky_imgs, axis=0)
    # I0_ref = I0_new / np.sum( I0_new ) 

    # strehl_signal[k] = np.mean( np.mean(lucky_ext_signals) )


    # img_list = [I0_ref.reshape(32,32), I0_unlucky_ref.reshape(32,32) ]
    # title_list = [f"Lucky I0\n{k}", f"Unlucky I0\n{k}"]
    # vlims = [[0,2*np.quantile(I0_ref,0.99 )], [0,2*np.quantile(I0_ref,0.99 )]]
    # util.nice_heatmap_subplots(im_list = img_list,
    #                         title_list = title_list,
    #                         vlims=vlims,
    #                         cbar_label_list = ['intensity [adu]','intensity [adu]'],)

print (strehl_signal)


