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


# TO DO 
# make a function to comapre, two different periods, compare time series section and PSDs 
# optimize gains (transfer function analysis)
# update toml (clear pupil, I0, inner_pupil_filter, N0_runtime, I2M_LO, I2M_HO) 
# argparser 



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

beam = 3
phasemask = "H4"
config_path = "/usr/local/etc/baldr/baldr_config_3.toml"
root_path = "/data/20260208"  # data/YYYYMMDD/


# 
# simplest input: strings "HH:MM:SS[.sss]"
#timestart = "01:08:00.000"
#timeend   = "01:09:30.000"

timestart = "01:06:50.000"
timeend   = "01:08:00.000"


# camera sampling assumptions (needed for PSD)
fs = 1000.0  # Hz (set from your actual camera FPS / burst rate)
max_ho_show = 8
ho_show = tuple(range(max_ho_show))  # indices to plot for HO


# ---------- load raw telemetry ----------

zwfs_telem = get_telemetry(root_path, instr="baldr", beam=beam, timestart=timestart, timeend=timeend)
#clear_telem = get_telemetry(root_path, instr="baldr", beam=beam, timestart=timestart, timeend=timeend)

if zwfs_telem.size == 0:
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

imgs = zwfs_telem  # (N, ny, nx)

# subtract dark (broadcast safe)
i_raw = imgs.astype(np.float32, copy=False) - dark.reshape(1, imgs.shape[1], imgs.shape[2])

if space == "pix":
    i_space = i_raw.reshape(i_raw.shape[0], -1)  # (N, P)
elif space == "dm":
    i_space = (I2A @ i_raw.reshape(i_raw.shape[0], -1).T).T  # (N, P')
else:
    raise ValueError(f"Unknown space: {space!r}")

# opd metric
opd_sig_tmp = np.mean(i_space[:, strehl_filt], axis=1) / N0_runtime
opd_metric = 0.03 * piecewise_continuous(opd_sig_tmp, interc, slope_1, slope_2, x_knee)

# normalized intensity & signal
i_norm = i_space / N0_runtime
s = i_norm - i_setpoint_runtime[None, :]

# modal errors (shapes: LO -> (2,N), HO -> (Nm,N))
e_LO = (I2M_LO @ s.T)
e_HO = (I2M_HO @ s.T)

# ---------- plots: timeseries + PSD ----------

N = s.shape[0]
t_s = np.arange(N) / fs

_plot_timeseries(t_s, e_LO, title="e_LO time series", labels=["LO0", "LO1"])

ho_idx = np.array([i for i in ho_show if 0 <= i < e_HO.shape[0]], dtype=int)
if ho_idx.size:
    _plot_timeseries(
        t_s,
        e_HO[ho_idx],
        title=f"e_HO time series (selected {ho_idx.tolist()})",
        labels=[f"HO{i}" for i in ho_idx.tolist()],
    )

_plot_psd_with_rcum(e_LO, fs=fs, title="e_LO Welch PSD (+ reverse cumulative)", labels=["LO0", "LO1"])

if ho_idx.size:
    _plot_psd_with_rcum(
        e_HO[ho_idx],
        fs=fs,
        title=f"e_HO Welch PSD (selected {ho_idx.tolist()}) (+ reverse cumulative)",
        labels=[f"HO{i}" for i in ho_idx.tolist()],
    )

plt.figure()
plt.plot(t_s, opd_metric)
plt.xlabel("time [s]")
plt.ylabel("OPD metric")
plt.title("OPD metric time series")
plt.grid(True, alpha=0.3)

plt.show()

### my skeleton 
# from astropy.io import fits
# import numpy as np
# import matplotlib.pyplot as plt
# import toml 
# from baldr_python_rtc.baldr_rtc.core.config import readBDRConfig
 
# # need to 
# # 1. populate get_files, also define best input type for timestart, timeend 
# # 2. optimize merge data in a way that is consistent with how I deal with it later in the script (numpy stack?)
# # 3. Produce time series and welch PSD of errors (e_LO, e_HO). Display them, LO has 2 modes, HO has upto 140 modes (so time series we could just seperate 1 plot for LO, 1 plot of HO, with selected inidicies for HO. Keep this all shortest and simplest possible code, don't ove complicate )
# #    PSD should also have a reverse cumulative dashed line   


# def get_files(root_path, beam, timestart, timeend ):
#     # assume root_path points to correct date 
#     # get fits files from Baldr/Heimdallr cred1 server telemetry 
#     # root_path/<instr><beam>_THH:MM.SSS.fits
#     # e.g. root_path/baldr3_T02:51:51.168.fits

#     # needs to return an ordered list of files 
#     # (from oldest to most recent timestamp in file name)
#     # between timestart and timeend. Not sure what best format input for these 'timestart and timeend' is?


# def merge_data( file_list ):
#     data_cube = []
#     for f in file_list:
#         d = fits.open(f)
#         data_cube.append( d[0].data ) #5000x32x32 np.array cube type int32
#     return data_cube


# def get_telemetry( root_path, timestart, timeend):
    
#     # some function to get list of telemetry files within timestamps
#     file_list = get_files(root_path, beam, timestart, timeend) 
#     for fname in file_list:
#         #d = fits.open( fname )
#         data_cube = merge_data( file_list )

#     return data_cube


# beam = 3
# phasemask = 'H4'
# config_path = '/usr/local/etc/baldr/baldr_config_3.toml' # baldr reconstructor config file 
# root_path = "/data/20260207" # include the date root_path = 'data/YYYYMMDD/'
# timestart = # string or datetime?
# timeend = # string or datetime?

# # cred 1 camera subframes 
# zwfs_telem = get_telemetry( root_path, beam, timestart, timeend)
# clear_telem = get_telemetry( root_path, beam, timestart, timeend)

# # baldr reconstructor configuration file 
# cfg = readBDRConfig(config_path=config_path, beam=beam, phasemask=phasemask)

# # useful shorthands we will use here (copied from baldr_python_rtc.baldr_rtc.server.build_rtc_model for consistency)
# state = cfg.state
# space = (state.signal_space or "pix").strip().lower()

# I2A = np.asarray(cfg.matrices.I2A, dtype=float) # if space == "dm" else None

# I2M_LO = np.asarray(cfg.matrices.I2M_LO, dtype=float)
# I2M_HO = np.asarray(cfg.matrices.I2M_HO, dtype=float)
# M2C_LO = np.asarray(cfg.matrices.M2C_LO, dtype=float)
# M2C_HO = np.asarray(cfg.matrices.M2C_HO, dtype=float)

# # filters 
# inner_pupil_filt = np.asarray( cfg.filters.inner_pupil_filt, dtype=bool).reshape(-1) 

# # NEW (NOT LEGACY)
# strehl_filt =  np.asarray( cfg.filters.strehl_filt, dtype=bool).reshape(-1) 

# # reduction 
# I0 = np.asarray(cfg.reference_pupils.I0, dtype=float).reshape(-1)
# N0 = np.asarray(cfg.reference_pupils.N0, dtype=float).reshape(-1)
# dark = np.asarray(cfg.reference_pupils.dark, dtype=float).reshape(-1)
# if space == "dm":
#     # NOTE: assumes I0/N0 are already in the SAME reduced pixel vector space as I2A expects
#     I0 = I2A @ I0
#     N0 = I2A @ N0
#     inner_pupil_filt = (I2A @ inner_pupil_filt).astype(bool)
#     strehl_filt = (I2A @ strehl_filt).astype(bool)
# i_setpoint_runtime = I0 / np.mean( N0[inner_pupil_filt]  ) 
# N0_runtime = np.mean( N0[inner_pupil_filt]  ) #N0



# # opd model (kind of hard coded parameters now (from asgard-alignment/calibration/build_strehl_model_v2.py))
# def piecewise_continuous(x, interc, slope_1, slope_2, x_knee):
#     # piecewise linear (hinge) model 
#     return interc + slope_1 * x + slope_2 * np.maximum(0.0, x - x_knee)


# interc = 9368.549647307767
# slope_1 = -5882.950106515396
# slope_2 = 4678.104756734429
# x_knee = 1.5324802815558276

# # process images using baldr cfg

# # example with zwfs_telem
# imgs = zwfs_telem.copy()

# # Copy and make sure consistent with what was done in baldr_python_rtc/baldr_rtc/rtc/loop.py
# i_raw = np.array( [i_raw - dark for i_raw in imgs] )

# if space.lower().strip() == 'pix':
#     i_space = np.array( [i_raw.reshape(-1) for i_raw in imgs] )
# elif space.lower().strip() == 'dm':
#     i_space = np.array( [I2A @ i_raw.reshape(-1) for i_raw in imgs] )


# opd_sig_tmp  = np.array( [np.mean( iii[ strehl_filt] ) / N0_runtime  for iii in i_space] )
# opd_metric = 0.03 * piecewise_continuous( opd_sig_tmp , interc, slope_1, slope_2, x_knee) 

# # normalized intensity 
# i_norm = i_space / N0_runtime 

# s = i_norm  - i_setpoint_runtime 

# # project intensity signal to error in modal space 
# e_LO = I2M_LO @ s 
# e_HO = I2M_HO @ s


