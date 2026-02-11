#!/usr/bin/env python3
import numpy as np
import zmq
import time
import toml
import os
import argparse
import datetime

from astropy.io import fits
from xaosim.shmlib import shm
from asgard_alignment.DM_shm_ctrl import dmclass
import common.DM_basis_functions as dmbases
from asgard_alignment import FLI_Cameras as FLI


# ---------------------- MDS (Controllino / stage server) ----------------------
def mds_connect(host: str, port: int = 5555, timeout_ms: int = 5000):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.connect(f"tcp://{host}:{port}")
    return ctx, sock

def mds_send(sock, msg: str) -> str:
    sock.send_string(msg)
    return sock.recv_string().strip()


# ---------------------- CLI ----------------------
parser = argparse.ArgumentParser(description="On-sky fast mode modulation + lock-in demod; only update N0_runtime from clear pupil.")
default_toml = os.path.join("/usr/local/etc/baldr/", "baldr_config_#.toml")

parser.add_argument("--toml_file", type=str, default=default_toml)
parser.add_argument("--beam_id", type=int, default=3)
parser.add_argument("--phasemask", type=str, default="H4")
parser.add_argument(
    "--basis_name",
    type=str,
    default="zernike", #zernike | zonal
    help="basis used to build interaction matrix (IM). zonal, zernike, zonal"
)
parser.add_argument("--global_camera_shm", type=str, default="/dev/shm/cred1.im.shm")
parser.add_argument("--mds_host", type=str, default="192.168.100.2")
parser.add_argument("--mds_port", type=int, default=5555)

parser.add_argument("--LO", type=int, default=2)

parser.add_argument("--signal_space", type=str, default="dm", help="'pix' or 'dm' (demod performed in that space)")
parser.add_argument("--mode_idx", type=int, default=0, help="index into modal_basis to modulate")
parser.add_argument("--amp", type=float, default=0.05, help="modulation amplitude in DM command units")
parser.add_argument("--f_mod", type=float, default=200.0, help="modulation frequency (Hz)")
parser.add_argument("--n_cycles", type=int, default=100, help="number of cycles to record")
parser.add_argument("--settle_cycles", type=int, default=5, help="cycles to discard at start")

#parser.add_argument("--no_samples_per_cmd", type=int, default=50, help="frames to average for clear pupil N0 measurement")
#parser.add_argument("--sleep_s", type=float, default=0.01, help="sleep between camera reads")

#parser.add_argument("--rel_offset_um", type=float, default=-200.0, help="relative offset (um) to move BMX/BMY for clear pupil")
parser.add_argument("--out_dir", type=str, default="/home/asg/Progs/repos/asgard-alignment/calibration/reports/modlockin")
parser.add_argument("--save_timeseries", action="store_true", help="store full time series (large)")

args = parser.parse_args()


# ---------------------- Load TOML + matrices / references ----------------------
toml_path = args.toml_file.replace("#", f"{args.beam_id}")
with open(toml_path, "r") as f:
    cfg = toml.load(f)

beam_key = f"beam{args.beam_id}"
ctrl = cfg.get(beam_key, {}).get(args.phasemask, {}).get("ctrl_model", {})
if not ctrl:
    raise RuntimeError(f"Missing ctrl_model for {beam_key}/{args.phasemask} in {toml_path}")

I2A = np.array(cfg[beam_key]["I2A"], dtype=np.float32)
I2M_LO = np.array(ctrl["I2M_LO"], dtype=np.float32)
I2M_HO = np.array(ctrl["I2M_HO"], dtype=np.float32)

I0 = np.array(ctrl["I0"], dtype=np.float32).reshape(-1)
N0 = np.array(ctrl["N0"], dtype=np.float32).reshape(-1)
dark = np.array(ctrl["dark"], dtype=np.float32).reshape(-1)

inner_pupil_filt = np.array(ctrl["inner_pupil_filt"])
# keep your existing lab-consistent setpoint definition
N0_runtime = float(np.mean(N0[inner_pupil_filt]))
i_setpoint_runtime = (I0 / N0_runtime).astype(np.float32)

if args.signal_space.lower().strip() not in ("pix", "dm"):
    raise ValueError("signal_space must be 'pix' or 'dm'")


# ---------------------- Camera + DM SHM ----------------------
camclient = FLI.fli(args.global_camera_shm, roi=[None, None, None, None])
camera_config_current = {k: str(v) for k, v in camclient.config.items()}

cam_shm = shm(f"/dev/shm/baldr{args.beam_id}.im.shm")
dm_shm = dmclass(beam_id=args.beam_id, main_chn=2)  # you said poke channel is handled in this object


# ---------------------- MDS connect + move to phasemask ----------------------
ctx, sock = mds_connect(args.mds_host, args.mds_port, timeout_ms=5000)

print(f"Moving to phasemask {args.phasemask} reference position...")
res = mds_send(sock, f"fpm_movetomask phasemask{args.beam_id} {args.phasemask}")
print(res)
time.sleep(1.0)


# ---------------------- Measure on-sky N0_runtime from clear pupil ----------------------
print("Moving FPM out to get clear pupil (for N0 measurement)...")
rel = -200 #float(args.rel_offset_um)
print(mds_send(sock, f"moverel BMX{args.beam_id} {rel}"))
time.sleep(0.5)
print(mds_send(sock, f"moverel BMY{args.beam_id} {rel}"))
time.sleep(1.0)

frames = []
for _ in range(int(200)):
    frames.append(cam_shm.get_data().reshape(-1) - dark.reshape(-1))
    time.sleep(float(0.01))
N0_measured_0 = np.mean(frames, axis=0).astype(np.float32)

if args.signal_space.lower().strip() == "dm":
    N0_measured = (I2A @ N0_measured_0).astype(np.float32)
else:
    N0_measured = N0_measured_0
    
N0_runtime_measured = float(np.mean(N0_measured.reshape(-1)[inner_pupil_filt.reshape(-1)]))

print(f"previous N0_runtime  = {N0_runtime:.6g}")
print(f"measured  N0_runtime  = {N0_runtime_measured:.6g}")

if (not np.isfinite(N0_runtime_measured)) or (N0_runtime_measured <= 0):
    raise RuntimeError(f"Bad measured N0_runtime: {N0_runtime_measured}")

ratio = N0_runtime_measured / N0_runtime if N0_runtime > 0 else np.nan
if np.isfinite(ratio) and (ratio < 0.01 or ratio > 100.0):
    print(f"WARNING: N0_runtime changed a lot (ratio={ratio:.2f}). Was the mask fully out?")

usr = input("Type 1 to ACCEPT this new N0_runtime (otherwise keep old): ").strip()
if usr == "1":
    N0_runtime = float(N0_runtime_measured)
    print(f"Updated N0_runtime -> {N0_runtime:.6g}")
else:
    print("Keeping previous N0_runtime.")

print("Moving FPM back in beam...")
print(mds_send(sock, f"moverel BMX{args.beam_id} {-rel}"))
time.sleep(0.5)
print(mds_send(sock, f"moverel BMY{args.beam_id} {-rel}"))
time.sleep(1.0)

input("Confirm mask is back in & aligned, then press ENTER to start modulation...")


# ---------------------- Build modal basis (same as your ramp script) ----------------------
LO_basis = dmbases.zer_bank(2, args.LO+1 )
if 'zonal' in args.basis_name.lower().strip():
    zonal_basis = np.array([dm_shm.cmd_2_map2D(ii) for ii in np.eye(140)]) 
elif 'zernike' in args.basis_name.lower().strip():
    zonal_basis = dmbases.zer_bank(4, 143 ) # ignore bad 'zonal_basis' naming
else:
    raise UserWarning(f'invalid --basis_name={args.basis_name} input. must be "zonal" or "zernike",')
modal_basis = np.array( LO_basis.tolist() +  zonal_basis.tolist() ) 


if args.mode_idx < 0 or args.mode_idx >= modal_basis.shape[0]:
    raise ValueError(f"mode_idx out of range: {args.mode_idx} / {modal_basis.shape[0]}")

mode_vec = modal_basis[args.mode_idx].astype(np.float32)


# ---------------------- Acquire modulation time series ----------------------
f = float(args.f_mod)
omega = 2.0 * np.pi * f
T_total = float(args.n_cycles) / f
T_settle = float(args.settle_cycles) / f

t_list = []
y_list = []
eLO_list = []
eHO_list = []

t0 = time.perf_counter()
last_print = time.perf_counter()

while True:
    t = time.perf_counter() - t0
    if t >= T_total:
        break

    inj = float(args.amp) * mode_vec * np.sin(omega * t)
    dm_shm.set_data(inj)

    fr = cam_shm.get_data().astype(np.float32)
    fr_flat = fr.reshape(-1) - dark

    # KEEP SAME SIGNAL DEFINITION, ONLY N0_runtime UPDATED ON-SKY
    signal_pix = (fr_flat / float(N0_runtime) - i_setpoint_runtime).astype(np.float32)

    if args.signal_space.lower().strip() == "dm":
        y = (I2A @ signal_pix).astype(np.float32)
    else:
        y = signal_pix

    t_list.append(t)
    y_list.append(y)

    # optional compact demod sanity telemetry (in pix definition)
    eLO_list.append((I2M_LO @ signal_pix).astype(np.float32))
    eHO_list.append((I2M_HO @ signal_pix).astype(np.float32))

    now = time.perf_counter()
    if now - last_print > 1.0:
        last_print = now
        print(f"t={t:.2f}s / {T_total:.2f}s")

# stop poke channel
dm_shm.set_data(0.0 * mode_vec)

t_arr = np.asarray(t_list, dtype=np.float64)
y_arr = np.asarray(y_list, dtype=np.float32)      # (N, P)
eLO_arr = np.asarray(eLO_list, dtype=np.float32)  # (N, n_lo)
eHO_arr = np.asarray(eHO_list, dtype=np.float32)  # (N, n_ho)

keep = t_arr >= T_settle
t_use = t_arr[keep]
y_use = y_arr[keep, :]
eLO_use = eLO_arr[keep, :]
eHO_use = eHO_arr[keep, :]

ref_sin = np.sin(omega * t_use).astype(np.float32)
ref_cos = np.cos(omega * t_use).astype(np.float32)

N = int(y_use.shape[0])
a = (2.0 / N) * (ref_sin[:, None] * y_use).sum(axis=0)
b = (2.0 / N) * (ref_cos[:, None] * y_use).sum(axis=0)
H = b + 1j * a

amp = np.abs(H).astype(np.float32)
phs = np.angle(H).astype(np.float32)

Nlo = int(eLO_use.shape[0])
a_lo = (2.0 / Nlo) * (ref_sin[:, None] * eLO_use).sum(axis=0)
b_lo = (2.0 / Nlo) * (ref_cos[:, None] * eLO_use).sum(axis=0)
H_lo = b_lo + 1j * a_lo

Nho = int(eHO_use.shape[0])
a_ho = (2.0 / Nho) * (ref_sin[:, None] * eHO_use).sum(axis=0)
b_ho = (2.0 / Nho) * (ref_cos[:, None] * eHO_use).sum(axis=0)
H_ho = b_ho + 1j * a_ho


# ---------------------- Write FITS ----------------------
tstamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
tstamp_rough = datetime.datetime.now().strftime("%Y-%m-%d")

out_dir = os.path.join(args.out_dir, tstamp_rough)
os.makedirs(out_dir, exist_ok=True)
fits_path = os.path.join(
    out_dir,
    f"modlockin_beam{args.beam_id}_{args.phasemask}_mode{args.mode_idx:04d}_{tstamp}.fits",
)

hdr = fits.Header()
hdr["DATE"] = tstamp
hdr["BEAMID"] = int(args.beam_id)
hdr["PHMASK"] = str(args.phasemask)
hdr["SIGSPC"] = str(args.signal_space)
hdr["MODEIDX"] = int(args.mode_idx)
hdr["AMP"] = float(args.amp)
hdr["FMOD"] = float(args.f_mod)
hdr["NCYCLE"] = int(args.n_cycles)
hdr["SETCYC"] = int(args.settle_cycles)
hdr["N0RUN0"] = float(np.mean(N0[inner_pupil_filt]))   # toml scalar
hdr["N0RUNC"] = float(N0_runtime)                       # used scalar
hdr["TOML"] = toml_path[:68]

hdus = [fits.PrimaryHDU(header=hdr)]
hdus.append(fits.ImageHDU(data=N0_measured.astype(np.float32), name="N0_MEAS"))
hdus.append(fits.ImageHDU(data=mode_vec.astype(np.float32), name="MODE_VEC"))

hdus.append(fits.ImageHDU(data=np.real(H).astype(np.float32), name="H_REAL"))
hdus.append(fits.ImageHDU(data=np.imag(H).astype(np.float32), name="H_IMAG"))
hdus.append(fits.ImageHDU(data=amp.astype(np.float32), name="H_AMP"))
hdus.append(fits.ImageHDU(data=phs.astype(np.float32), name="H_PHS"))

hdus.append(fits.ImageHDU(data=np.real(H_lo).astype(np.float32), name="HLO_REAL"))
hdus.append(fits.ImageHDU(data=np.imag(H_lo).astype(np.float32), name="HLO_IMAG"))
hdus.append(fits.ImageHDU(data=np.real(H_ho).astype(np.float32), name="HHO_REAL"))
hdus.append(fits.ImageHDU(data=np.imag(H_ho).astype(np.float32), name="HHO_IMAG"))

if args.save_timeseries:
    hdus.append(fits.ImageHDU(data=t_arr.astype(np.float64), name="T"))
    hdus.append(fits.ImageHDU(data=y_arr.astype(np.float32), name="Y"))
    hdus.append(fits.ImageHDU(data=eLO_arr.astype(np.float32), name="E_LO_TS"))
    hdus.append(fits.ImageHDU(data=eHO_arr.astype(np.float32), name="E_HO_TS"))

cam_keys = np.array(list(camera_config_current.keys()), dtype="S64")
cam_vals = np.array([str(v) for v in camera_config_current.values()], dtype="S128")
cam_cols = [
    fits.Column(name="KEY", format="64A", array=cam_keys),
    fits.Column(name="VALUE", format="128A", array=cam_vals),
]
hdus.append(fits.BinTableHDU.from_columns(cam_cols, name="CAMCFG_CURR"))

toml_dump = toml.dumps(cfg)
toml_lines = np.array(toml_dump.splitlines(), dtype="S200")
toml_col = fits.Column(name="TOML", format="200A", array=toml_lines)
hdus.append(fits.BinTableHDU.from_columns([toml_col], name="TOML_DUMP"))

fits.HDUList(hdus).writeto(fits_path, overwrite=True)
print(f"Wrote FITS: {fits_path}")