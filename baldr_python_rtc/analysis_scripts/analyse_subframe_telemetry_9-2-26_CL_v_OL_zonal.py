from __future__ import annotations

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from datetime import datetime, date, time
from scipy.signal import welch

from baldr_python_rtc.baldr_rtc.core.config import readBDRConfig

# Note: timestampe cred teleemtry is UT time, goes to new date at 00:00 


###############################################
# NOTES: 
###############################################
# close vs open loop HO/TT beam 2 mask H4 with zonal basis (all modes) 
# we also had heimdallr shutters up apart from beam 2 so 
#   to do : we SHOULD look at the heimdallr frames too! 
# timing check of rtc at 1kHz frame rate was ~ 0.001ms (1000Hz), 
# averaged 1 frames before applying correction (in loop.py we set no_2_avg = 3)
# seeing was quiet bad (~1.2")
# TT gain : u_LO = 0 * u_LO - 0 * e_LO 
# HO gain : u_HO = 0.97 * u_HO - 0.08 * e_LO something like this 
# result - good




# ---------- telemetry I/O ----------

# pattern for matching time in cred 1 files
_TIME_RE = re.compile(r"_T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?\.fits$")

def _parse_hms(x) -> float:
    """
    Accepts:
      - "HH:MM:SS[.sss]" string
      - datetime.time
      - datetime.datetime (uses time-of-day)
      - float/int (seconds since 00:00)
    Returns seconds since midnight.
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, datetime):
        x = x.time()
    if isinstance(x, time):
        return x.hour * 3600.0 + x.minute * 60.0 + x.second + x.microsecond * 1e-6
    if isinstance(x, str):
        x = x.strip()
        try:
            # allow "HH:MM:SS.sss" -> time
            t = datetime.strptime(x, "%H:%M:%S.%f").time()
        except ValueError:
            t = datetime.strptime(x, "%H:%M:%S").time()
        return t.hour * 3600.0 + t.minute * 60.0 + t.second + t.microsecond * 1e-6
    raise TypeError(f"Unsupported time type: {type(x)!r}")


def _file_time_seconds(path: Path) -> float | None:
    m = _TIME_RE.search(path.name)
    if not m:
        return None
    hh, mm, ss, frac = m.groups()
    us = int((frac or "0").ljust(6, "0")[:6])
    return int(hh) * 3600.0 + int(mm) * 60.0 + int(ss) + us * 1e-6


def get_files(root_path, instr: str, beam: int, timestart, timeend):
    """
    root_path: ".../YYYYMMDD"
    instr: "baldr" or "heimdallr" etc. (prefix used in filename)
    beam: integer (e.g. 3)
    timestart/timeend: "HH:MM:SS[.sss]" | datetime.time | datetime.datetime | seconds(float)
    Returns sorted list of Path (oldest->newest) within [timestart, timeend].
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"root_path not found: {root}")

    t0 = _parse_hms(timestart)
    t1 = _parse_hms(timeend)
    if t1 < t0:
        raise ValueError("timeend < timestart (expected same-day window)")

    pat = f"{instr}{int(beam)}_T*.fits"
    cand = sorted(root.glob(pat))

    keep = []
    for p in cand:
        ts = _file_time_seconds(p)
        if ts is None:
            continue
        if t0 <= ts <= t1:
            keep.append((ts, p))

    keep.sort(key=lambda x: x[0])
    return [p for _, p in keep]


def merge_data(file_list, hdu=0, dtype=np.float32):
    """
    Concatenate telemetry cubes along time axis.
    Each file assumed shape (n, ny, nx). Returns array (N, ny, nx).
    """
    if not file_list:
        return np.empty((0, 0, 0), dtype=dtype)

    chunks = []
    ny = nx = None
    for f in file_list:
        try:
            d = fits.getdata(f, ext=hdu, memmap=True)
        except Exception as e:
            print(f"[merge_data] skip unreadable file: {f} ({e})")
            continue

        if d.ndim != 3:
            print(f"[merge_data] skip (expected 3D cube): {f} shape={getattr(d, 'shape', None)}")
            continue

        if ny is None:
            _, ny, nx = d.shape
        elif d.shape[1:] != (ny, nx):
            print(f"[merge_data] skip (shape mismatch): {f} shape={d.shape} expected=(*,{ny},{nx})")
            continue

        chunks.append(np.asarray(d, dtype=dtype))

    if not chunks:
        return np.empty((0, 0, 0), dtype=dtype)

    return np.concatenate(chunks, axis=0)


def get_telemetry(root_path, instr: str, beam: int, timestart, timeend):
    files = get_files(root_path, instr=instr, beam=beam, timestart=timestart, timeend=timeend)
    return merge_data(files)


# ---------- plotting helpers ----------

def _plot_timeseries(t_s, y, title, labels=None):
    plt.figure()
    if y.ndim == 1:
        plt.plot(t_s, y)
    else:
        for k in range(y.shape[0]):
            plt.plot(t_s, y[k], label=None if labels is None else labels[k])
        if labels is not None:
            plt.legend(loc="best", fontsize=9)
    plt.xlabel("time [s]")
    plt.ylabel("error")
    plt.title(title)
    plt.grid(True, alpha=0.3)


def _plot_psd_with_rcum(y, fs, title, labels=None, nperseg=None):
    """
    y: (modes, N) or (N,)
    Plots PSD plus reverse cumulative integral (dashed), per curve.
    """
    plt.figure()
    if y.ndim == 1:
        y = y[None, :]

    for k in range(y.shape[0]):
        f, Pxx = welch(y[k], fs=fs, nperseg=nperseg, detrend="constant", scaling="density")
        df = float(f[1] - f[0]) if f.size > 1 else 1.0
        rc = np.cumsum(Pxx[::-1]) * df
        rc = rc[::-1]

        lab = None if labels is None else labels[k]
        plt.loglog(f[1:], Pxx[1:], label=lab)
        plt.loglog(f[1:], rc[1:], linestyle="--", alpha=0.8)

    if labels is not None:
        plt.legend(loc="best", fontsize=9)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD [err^2/Hz]  (dashed: reverse cumulative integral PSD df)")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)


# ---------- user parameters ----------

beam = 2
phasemask = "H4"
config_path = "/home/asg/ben_feb2026_bld_telem/2026-02-09/baldr_config_2_2026-02-09T02-51-56_HO_lock_zonal_basis_all_modes.toml"
root_path = "/data/20260209" #"/data/20260208"  # data/YYYYMMDD/





# simplest input: strings "HH:MM:SS[.sss]"
timestart_CL = "05:25:50.000"#"05:31:50.000"
timeend_CL   = "05:26:50.000"#"05:35:00.000"

timestart_OL = "05:26:00.000"
timeend_OL   = "05:28:00.000"


# camera sampling assumptions (needed for PSD)
fs = 1000.0  # Hz (set from your actual camera FPS / burst rate)
max_ho_show = 8
ho_show = tuple(range(max_ho_show))  # indices to plot for HO


# ---------- load raw telemetry ----------

OL_telem = get_telemetry(root_path, instr="baldr", beam=beam, timestart=timestart_OL, timeend=timeend_OL)
CL_telem = get_telemetry(root_path, instr="baldr", beam=beam, timestart=timestart_CL, timeend=timeend_CL)

if OL_telem.size == 0:
    raise RuntimeError("No ZWFS telemetry loaded for the requested window")


# ---------- load baldr config + shorthands (match rtc conventions) ----------

cfg = readBDRConfig(config_path=config_path, beam=beam, phasemask=phasemask)
state = cfg.state
space = (state.signal_space or "pix").strip().lower()

I2A = np.asarray(cfg.matrices.I2A, dtype=float)

I2M_LO = np.asarray(cfg.matrices.I2M_LO, dtype=float)
I2M_HO = np.asarray(cfg.matrices.I2M_HO, dtype=float)
M2C_LO = np.asarray(cfg.matrices.M2C_LO, dtype=float)
M2C_HO = np.asarray(cfg.matrices.M2C_HO, dtype=float)

inner_pupil_filt = np.asarray(cfg.filters.inner_pupil_filt, dtype=bool).reshape(-1)
strehl_filt = np.asarray(cfg.filters.strehl_filt, dtype=bool).reshape(-1)

I0 = np.asarray(cfg.reference_pupils.I0, dtype=float).reshape(-1)
N0 = np.asarray(cfg.reference_pupils.N0, dtype=float).reshape(-1)
dark = np.asarray(cfg.reference_pupils.dark, dtype=float).reshape(-1)

if space == "dm":
    I0 = I2A @ I0
    N0 = I2A @ N0
    inner_pupil_filt = (I2A @ inner_pupil_filt).astype(bool)
    strehl_filt = (I2A @ strehl_filt).astype(bool)

N0_runtime = float(np.mean(N0[inner_pupil_filt]))
i_setpoint_runtime = I0 / N0_runtime


# ---------- opd metric model ----------

def piecewise_continuous(x, interc, slope_1, slope_2, x_knee):
    return interc + slope_1 * x + slope_2 * np.maximum(0.0, x - x_knee)

interc = 9368.549647307767
slope_1 = -5882.950106515396
slope_2 = 4678.104756734429
x_knee = 1.5324802815558276


# ---------- processing (ZWFS example) ----------
res = {}
for lab,imgs in zip(['OL','CL'],[OL_telem, CL_telem]):
    
    res[lab] = {}
    # subtract dark (broadcast safe)
    res[lab]['i_raw'] = imgs.astype(np.float32, copy=False) - dark.reshape(1, imgs.shape[1], imgs.shape[2])

    if space == "pix":
        res[lab]['i_space'] = res[lab]['i_raw'].reshape(res[lab]['i_raw'].shape[0], -1)  # (N, P)
    elif space == "dm":
        res[lab]['i_space'] = (I2A @ res[lab]['i_raw'].reshape(res[lab]['i_raw'].shape[0], -1).T).T  # (N, P')
    else:
        raise ValueError(f"Unknown space: {space!r}")

    # opd metric
    opd_sig_tmp = np.mean(res[lab]['i_space'][:, strehl_filt], axis=1) / N0_runtime
    res[lab]['opd_metric'] = 0.03 * piecewise_continuous(opd_sig_tmp, interc, slope_1, slope_2, x_knee)

    # normalized intensity & signal
    res[lab]['i_norm'] = res[lab]['i_space'] / N0_runtime
    res[lab]['s'] = res[lab]['i_norm'] - i_setpoint_runtime[None, :]

    # modal errors (shapes: LO -> (2,N), HO -> (Nm,N))
    res[lab]['e_LO'] = (I2M_LO @ res[lab]['s'].T)
    res[lab]['e_HO'] = (I2M_HO @ res[lab]['s'].T)


# ---------- plots: timeseries + PSD ----------

N = res[lab]['s'].shape[0]
t_s = np.arange(N) / fs



plt.figure()
for lab,col in zip(res,['k','r']):
    y = res[lab]['e_LO']
    if y.ndim == 1:
        y = y[None, :]

    for k in range(y.shape[0]):
        f, Pxx = welch(y[k], fs=fs, nperseg=None, detrend="constant", scaling="density")
        df = float(f[1] - f[0]) if f.size > 1 else 1.0
        rc = np.cumsum(Pxx[::-1]) * df
        rc = rc[::-1]

        #lab = None if labels is None else labels[k]
        plt.loglog(f[1:], Pxx[1:], label=lab,color=col)
        plt.loglog(f[1:], rc[1:], linestyle="--",color=col, alpha=0.8)

plt.legend(loc="best", fontsize=9)
plt.xlabel("frequency [Hz]")
plt.ylabel("PSD [err^2/Hz]\nrev. cum [err^2]")
#plt.title(title)
plt.grid(True, which="both", alpha=0.3)
plt.show()



####################### CORRELATION ANALYSIS 

# ---------- TT/HO correlation analysis ----------

def _zscore_rows(x, eps=1e-12):
    # x: (modes, N)
    x = np.asarray(x, float)
    mu = np.nanmean(x, axis=1, keepdims=True)
    xc = x - mu
    sig = np.nanstd(xc, axis=1, keepdims=True)
    sig = np.maximum(sig, eps)
    return xc / sig

def _corr_tt_ho(e_lo, e_ho, zscore=True):
    """
    e_lo: (2, N)
    e_ho: (Nho, N)
    returns:
      C: (2, Nho) Pearson corr
      valid: (N,) boolean mask actually used
    """
    e_lo = np.asarray(e_lo, float)
    e_ho = np.asarray(e_ho, float)
    if e_lo.ndim != 2 or e_ho.ndim != 2:
        raise ValueError("expected e_lo (2,N), e_ho (Nho,N)")
    if e_lo.shape[1] != e_ho.shape[1]:
        raise ValueError("time length mismatch")

    # robust mask: drop any time sample with NaN/Inf in any used channel
    x = e_lo
    y = e_ho
    ok = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)

    x = x[:, ok]
    y = y[:, ok]

    if zscore:
        x = _zscore_rows(x)
        y = _zscore_rows(y)
    else:
        # still demean so correlation is meaningful if you later swap in dot-products
        x = x - np.nanmean(x, axis=1, keepdims=True)
        y = y - np.nanmean(y, axis=1, keepdims=True)

    # With zscore: corr = (x @ y.T)/(N-1)
    # x: (2,N), y:(Nho,N) -> (2,Nho)
    N = x.shape[1]
    denom = max(N - 1, 1)
    C = (x @ y.T) / denom
    C = np.clip(C, -1.0, 1.0)
    return C, ok

def _sym_lims(C_list, q=99.0):
    # symmetric color limits using a percentile across multiple matrices
    v = np.concatenate([np.ravel(np.abs(C)) for C in C_list if C.size])
    if v.size == 0:
        return 1.0
    return float(np.nanpercentile(v, q))

def _plot_corr_summary(res, top_k=12, title_prefix="TT–HO corr", ho_limit=None):
    # compute correlations
    C = {}
    ok = {}
    for lab in res:
        C[lab], ok[lab] = _corr_tt_ho(res[lab]["e_LO"], res[lab]["e_HO"], zscore=True)

    # consistent color scaling for OL vs CL
    clim = _sym_lims([C["OL"], C["CL"]]) if ("OL" in C and "CL" in C) else _sym_lims(list(C.values()))

    # optionally truncate HO modes for plotting
    def _slice(Cmat):
        if ho_limit is None:
            return Cmat
        return Cmat[:, :int(ho_limit)]

    # 1) heatmaps
    plt.figure(figsize=(11, 4))
    for i, lab in enumerate(["OL", "CL"]):
        if lab not in C:
            continue
        plt.subplot(1, 2, i + 1)
        M = _slice(C[lab])
        im = plt.imshow(M, aspect="auto", origin="lower", vmin=-clim, vmax=clim)
        plt.yticks([0, 1], ["TT0", "TT1"])
        plt.xlabel("HO mode index")
        plt.title(f"{title_prefix} ({lab})  N={ok[lab].sum()}")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="corr")
    plt.tight_layout()

    # 2) |corr| summary vs HO index (max over TT)
    plt.figure(figsize=(11, 4))
    for lab, ls in zip(["OL", "CL"], ["-", "--"]):
        if lab not in C:
            continue
        M = _slice(C[lab])
        absM = np.abs(M)
        best = np.nanmax(absM, axis=0)             # (Nho,)
        which = np.nanargmax(absM, axis=0)         # 0 or 1 per HO
        idx = np.arange(best.size)
        plt.plot(idx, best, linestyle=ls, label=f"{lab}: max |corr|")
        # annotate TT assignment lightly as a stepped line near top
        plt.plot(idx, 0.02 + 0.02 * which, linestyle=ls, alpha=0.6)

    plt.xlabel("HO mode index")
    plt.ylabel("max |corr(TT, HO)|")
    plt.title(f"{title_prefix} summary (max over TT); small offset line indicates TT0 vs TT1")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    # 3) top-k scatter sanity checks for each dataset
    for lab in ["OL", "CL"]:
        if lab not in C:
            continue
        M = _slice(C[lab])
        absM = np.abs(M)
        score = np.nanmax(absM, axis=0)
        order = np.argsort(score)[::-1]
        order = order[: min(top_k, order.size)]

        e_lo = np.asarray(res[lab]["e_LO"], float)
        e_ho = np.asarray(res[lab]["e_HO"], float)
        # same mask used in corr
        m = ok[lab]
        e_lo = e_lo[:, m]
        e_ho = e_ho[:, m]

        plt.figure(figsize=(11, 6))
        ncols = 4
        nrows = int(np.ceil(order.size / ncols))
        for j, ho_idx in enumerate(order):
            tt_idx = int(np.nanargmax(np.abs(M[:, ho_idx])))
            r = float(M[tt_idx, ho_idx])

            plt.subplot(nrows, ncols, j + 1)
            plt.plot(e_lo[tt_idx], e_ho[ho_idx], ".", markersize=2, alpha=0.5)
            plt.title(f"{lab} HO{ho_idx} vs TT{tt_idx}\nr={r:+.2f}")
            plt.xlabel(f"TT{tt_idx}")
            plt.ylabel(f"HO{ho_idx}")
            plt.grid(True, alpha=0.2)

        plt.tight_layout()

    plt.show()

# run it
_plot_corr_summary(res, top_k=12, title_prefix="TT–HO correlation", ho_limit=None)