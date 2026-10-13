#!/usr/bin/env python3
"""Measure the Baldr DM -> ZWFS intensity response to 2-D sinusoidal probes.

Examples
--------
# Print the scan without opening shared memory:
python baldr_dm_spatial_transfer_acquire.py --beam 1

# Run the default scan on DM channel 3:
python baldr_dm_spatial_transfer_acquire.py --beam 1 --output transfer_b1.fits

# Short initial test:
python baldr_dm_spatial_transfer_acquire.py --beam 1  \
    --angles 0:180:45 --frequencies 1:4:1 --phases 0 90 180 270 \
    --rms 0.03 --frames 10 --camera-semid 2 --output transfer_test.fits

Angle is the DM spatial-frequency-vector angle, not necessarily the detector
angle. Frequencies are in cycles per nominal illuminated pupil (10 actuator
pitches). Phase is relative to the centre of the DM array.

Requires: numpy, astropy, xaosim.shmlib.
No mask movement, DM flat updates, or RTC commands are performed.
Camera frames are acquired by blocking on a dedicated camera semaphore, not
polling the image counter. Use a producer-posted semaphore not used by the RTC;
camera semaphore 1 is reserved for the RTC on the existing Baldr setup.
Note: xaosim blocking semaphore reads have no built-in timeout.
"""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np


DM_SIZE = 12
PUPIL_DIAMETER = 10.0        # Actuator pitches across illuminated pupil
DM_UPDATE_SEMID = 1          # Post on summed/master stream after channel write


def parse_range(value):
    """Accept 'start:stop:step' (exclusive stop) or comma-separated values."""
    if ':' in value:
        start, stop, step = map(float, value.split(':'))
        if step <= 0 or stop <= start:
            raise ValueError('Expected start:stop:positive_step with stop > start')
        return np.arange(start, stop - step * 1e-8, step).tolist()
    return [float(v) for v in value.split(',')]


def geometry():
    """DM coordinates in units of illuminated-pupil diameter."""
    row, col = np.indices((DM_SIZE, DM_SIZE))
    x = (col - 5.5) / PUPIL_DIAMETER
    y = (row - 5.5) / PUPIL_DIAMETER
    active = np.ones((DM_SIZE, DM_SIZE), dtype=bool)
    active[[0, 0, -1, -1], [0, -1, 0, -1]] = False
    pupil = active & (x*x + y*y <= 0.25)
    return x, y, active, pupil


def make_probe(frequency, angle_deg, phase_deg, rms, x, y, active, pupil):
    """Zero-piston sine wave; requested RMS is measured within the pupil."""
    theta, phase = np.deg2rad([angle_deg, phase_deg])
    distance = x*np.cos(theta) + y*np.sin(theta)
    pattern = np.cos(2*np.pi*frequency*distance + phase)
    pattern -= pattern[pupil].mean()
    sigma = np.sqrt(np.mean(pattern[pupil]**2))
    if sigma < 1e-8:  # Some phases are invisible at the DM Nyquist frequency
        return None
    pattern *= rms / sigma
    pattern[~active] = 0.0
    return pattern


def scan_states(args):
    """Insert an unprobed reference at the beginning, periodically and at end."""
    states = [dict(kind='REF', angle=np.nan, freq=np.nan, phase=np.nan, rms=0.)]
    number = 0
    for cycle in range(args.cycles):
        for angle in args.angles:
            for frequency in args.frequencies:
                for rms in args.rms:
                    for phase in args.phases:
                        states.append(dict(kind='PROBE', cycle=cycle, angle=angle,
                                           freq=frequency, phase=phase, rms=rms))
                        number += 1
                        if args.reference_every and number % args.reference_every == 0:
                            states.append(dict(kind='REF', angle=np.nan, freq=np.nan,
                                               phase=np.nan, rms=0.))
    if states[-1]['kind'] != 'REF':
        states.append(dict(kind='REF', angle=np.nan, freq=np.nan, phase=np.nan, rms=0.))
    return states


def current_camera_frame(camera):
    """A Baldr stream can be a 2-D frame or a 3-D circular image buffer."""
    if camera.mtdata['naxis'] == 3:
        return camera.get_latest_data_slice(semid=None).copy()
    return camera.get_data().copy()


def read_fresh_frames(camera, semid, n, discard):
    """Wait for new camera frames using a dedicated xaosim SHM semaphore.

    Catch up once per DM state, to avoid collecting frames from the previous
    state. Each subsequent call blocks until the camera posts a semaphore.
    """
    camera.catch_up_with_sem(semid)
    frames, counters, times_ns = [], [], []
    last = camera.get_counter()

    seen = 0
    while len(frames) < n:
        if camera.mtdata['naxis'] == 3:
            frame = camera.get_latest_data_slice(semid=semid).copy()
        else:
            frame = camera.get_latest_data(semid=semid).copy()
        counter = camera.get_counter()
        if counter < last:
            raise RuntimeError('Camera counter reset during acquisition')
        if counter == last:
            continue  # A queued semaphore post can refer to an already-read frame.
        last = counter
        seen += 1
        if seen <= discard:
            continue
        frames.append(np.asarray(frame, dtype=np.float32))
        counters.append(int(counter))
        times_ns.append(time.time_ns())
    return np.stack(frames), counters, times_ns


def prepare_fits(path, args, states, frame_shape, baseline, combined, initial_flat, pupil):
    """Preallocate one FITS file; each state is checkpointed in place."""
    from astropy.io import fits

    n = len(states)
    ny, nx = frame_shape
    settings = dict(beam=args.beam, channel=args.channel,
                    angles_deg=args.angles, frequencies_cpp=args.frequencies,
                    phases_deg=args.phases, amplitudes_rms=args.rms,
                    cycles=args.cycles, frames=args.frames,
                    nominal_pupil_diameter_actuators=PUPIL_DIAMETER,
                    coordinate_convention='x=DM columns, y=DM rows; 0 deg = +x; phase centre=(5.5,5.5)',
                    normalization='piston removed and RMS set over nominal circular 10-pitch pupil',
                    shm_dir=str(args.shm_dir.resolve()),
                    settle_seconds=args.settle_s, discard_frames=args.discard,
                    camera_semaphore=args.camera_semid,
                    reference_every=args.reference_every,
                    headroom_margin=args.headroom_margin,
                    created_utc=datetime.now(timezone.utc).isoformat())
    packed = json.dumps(settings)
    state_table = fits.BinTableHDU.from_columns([
        fits.Column(name='STEP', format='J', array=np.arange(n, dtype=np.int32)),
        fits.Column(name='KIND', format='5A', array=[s['kind'] for s in states]),
        fits.Column(name='CYCLE', format='J', array=[s.get('cycle', -1) for s in states]),
        fits.Column(name='ANGLE_DEG', format='E', array=[s['angle'] for s in states]),
        fits.Column(name='FREQ_CPP', format='E', array=[s['freq'] for s in states]),
        fits.Column(name='PHASE_DEG', format='E', array=[s['phase'] for s in states]),
        fits.Column(name='REQ_RMS', format='E', array=[s['rms'] for s in states]),
        fits.Column(name='ACT_RMS', format='E', array=np.full(n, np.nan, np.float32)),
        fits.Column(name='SCALE', format='E', array=np.full(n, np.nan, np.float32)),
        fits.Column(name='PEAK', format='E', array=np.full(n, np.nan, np.float32)),
        fits.Column(name='CMD_TIME_NS', format='K', array=np.zeros(n, dtype=np.int64)),
        fits.Column(name='STATUS', format='I', array=np.zeros(n, dtype=np.int16)),
        fits.Column(name='NOTE', format='64A', array=['']*n),
    ], name='STATES')

    header = fits.Header()
    header['BEAM'] = args.beam
    header['CHANNEL'] = args.channel
    header['PUPDIAM'] = PUPIL_DIAMETER
    header['COMPLETE'] = False
    header['RESTORED'] = False
    header['NSTATES'] = n
    header['NFRAMES'] = args.frames
    header['BUNIT'] = 'DM normalized command'
    hdus = fits.HDUList([
        fits.PrimaryHDU(header=header),
        fits.BinTableHDU.from_columns([
            fits.Column(name='JSON', format=f'{len(packed)}A', array=[packed])
        ], name='SETTINGS'),
        state_table,
        fits.ImageHDU(pupil.astype(np.uint8), name='PUPIL_DM'),
        fits.ImageHDU(baseline, name='CHANNEL_BEFORE'),
        fits.ImageHDU(combined, name='DM_BEFORE'),
        fits.ImageHDU(initial_flat, name='FLAT_BEFORE'),
        fits.ImageHDU(np.zeros((n, 12, 12), np.float32), name='DM_PROBES'),
        fits.ImageHDU(np.zeros((n, 12, 12), np.float32), name='DM_COMMANDS'),
        fits.ImageHDU(np.full((n, args.frames, ny, nx), np.nan, np.float32),
                      name='ZWFS_FRAMES'),
        fits.ImageHDU(np.full((n, ny, nx), np.nan, np.float32), name='ZWFS_MEANS'),
        fits.ImageHDU(np.zeros((n, args.frames), np.int64), name='CAM_COUNTERS'),
        fits.ImageHDU(np.zeros((n, args.frames), np.int64), name='CAM_TIMES_NS'),
    ])
    hdus.writeto(path, overwrite=args.overwrite)
    hdus.close()
    return fits.open(path, mode='update', memmap=True)


def acquire(args, states):
    from xaosim.shmlib import shm

    x, y, active, pupil = geometry()
    paths = dict(probe=args.shm_dir/f'dm{args.beam}disp{args.channel:02d}.im.shm',
                 master=args.shm_dir/f'dm{args.beam}.im.shm',
                 camera=args.shm_dir/f'baldr{args.beam}.im.shm',
                 flat=args.shm_dir/f'dm{args.beam}disp00.im.shm')
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'{args.output} exists (use --overwrite if intended)')
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        streams = {}
        for name, path in paths.items():
            # Both master and camera need their semaphores.
            stream = shm(str(path), nosem=(name not in ('master', 'camera')))
            stack.callback(stream.close, erase_file=False)
            for sem in getattr(stream, 'sems', []):
                stack.callback(sem.close)
            streams[name] = stream

        probe, master, camera, flat = (streams[k] for k in ('probe', 'master', 'camera', 'flat'))
        if camera.sems and args.camera_semid >= len(camera.sems):
            raise ValueError(f'Camera semaphore {args.camera_semid} unavailable '
                             f'({len(camera.sems)} semaphores exposed)')
        initial = probe.get_data().copy()
        initial_combined = master.get_data().copy()
        initial_flat = flat.get_data().copy()
        if initial.shape != (12, 12) or initial.dtype.kind != 'f':
            raise ValueError('Expected the probe channel to be a float 12x12 DM map')
        if initial_combined.shape != (12, 12) or initial_flat.shape != (12, 12):
            raise ValueError('Unexpected DM master or flat shape')
        frame_shape = current_camera_frame(camera).shape
        if len(frame_shape) != 2:
            raise ValueError(f'Expected a 2D Baldr subframe; received {frame_shape}')
        if not np.isfinite(initial_combined[active]).all():
            raise ValueError('Combined DM has non-finite actuator values')

        out = stack.enter_context(prepare_fits(args.output, args, states, frame_shape,
                                               initial, initial_combined, initial_flat, pupil))
        last_command = initial.copy()
        completed = False
        restored = False
        try:
            for i, s in enumerate(states):
                # Keep ownership of channel 3 (or the channel explicitly selected).
                if not np.array_equal(probe.get_data(), last_command):
                    raise RuntimeError('Probe channel changed externally; stopping')
                if not np.array_equal(flat.get_data(), initial_flat):
                    raise RuntimeError('DM flat changed externally; stopping')

                delta = np.zeros((12, 12), dtype=float)
                if s['kind'] == 'PROBE':
                    delta = make_probe(s['freq'], s['angle'], s['phase'], s['rms'],
                                       x, y, active, pupil)
                    if delta is None:
                        out['STATES'].data['STATUS'][i] = 2  # skipped: DM sampling null
                        out['STATES'].data['NOTE'][i] = 'DM Nyquist/sampling null'
                        print(f'{i+1}/{len(states)} skipped: sampled sinusoid is null', flush=True)
                        continue

                # The current combined stream may also contain other live DM channels.
                # Remove OUR previous perturbation to estimate its current baseline.
                summed = master.get_data().astype(float)
                base = summed - (last_command.astype(float) - initial)
                if np.any(base[active] <= args.headroom_margin) or np.any(
                        base[active] >= 1.0 - args.headroom_margin):
                    raise RuntimeError('Combined DM baseline has an actuator at/near a limit')

                # Scale the whole sine wave (not individual actuators) if headroom
                # is insufficient, preserving its shape and recording actual RMS.
                scale = 1.0
                positive = active & (delta > 0)
                negative = active & (delta < 0)
                if np.any(positive):
                    scale = min(scale, float(np.min(
                        (1.0 - args.headroom_margin - base[positive]) / delta[positive])))
                if np.any(negative):
                    scale = min(scale, float(np.min(
                        (base[negative] - args.headroom_margin) / -delta[negative])))
                scale = max(0.0, min(1.0, scale))
                if s['kind'] == 'PROBE' and scale < 1e-4:
                    raise RuntimeError('No DM headroom to apply the requested probe')
                delta *= scale
                command = (initial + delta).astype(initial.dtype)
                actual_delta = command.astype(float) - initial.astype(float)
                if not np.isfinite(command).all():
                    raise ValueError('Non-finite DM command')

                old_master_counter = master.get_counter()
                probe.set_data(command)
                last_command = command.copy()
                master.post_sems(DM_UPDATE_SEMID)
                command_time = time.time_ns()

                deadline = time.monotonic() + args.timeout
                while master.get_counter() == old_master_counter:
                    if time.monotonic() > deadline:
                        raise TimeoutError('DM master did not update after semaphore post')
                    time.sleep(0.002)
                summed_now = master.get_data()
                if np.any(summed_now[active] <= 0) or np.any(summed_now[active] >= 1):
                    raise RuntimeError('Combined DM saturated after the probe command')
                time.sleep(args.settle_s)

                frames, counters, times_ns = read_fresh_frames(
                    camera, args.camera_semid, args.frames, args.discard)
                if not np.array_equal(probe.get_data(), last_command):
                    raise RuntimeError('Another writer changed the probe channel')
                if not np.array_equal(flat.get_data(), initial_flat):
                    raise RuntimeError('DM flat changed during acquisition')
                if frames.shape[1:] != frame_shape:
                    raise RuntimeError('Camera frame shape changed during acquisition')

                out['DM_PROBES'].data[i] = actual_delta
                out['DM_COMMANDS'].data[i] = command
                out['ZWFS_FRAMES'].data[i] = frames
                out['ZWFS_MEANS'].data[i] = frames.mean(axis=0)
                out['CAM_COUNTERS'].data[i] = counters
                out['CAM_TIMES_NS'].data[i] = times_ns
                table = out['STATES'].data
                table['ACT_RMS'][i] = np.sqrt(np.mean(actual_delta[pupil]**2))
                table['SCALE'][i] = scale
                table['PEAK'][i] = np.max(np.abs(actual_delta[active]))
                table['CMD_TIME_NS'][i] = command_time
                table['STATUS'][i] = 1  # successful measurement
                table['NOTE'][i] = 'scaled for headroom' if scale < 0.9999 else ''

                if (i+1) % args.checkpoint_every == 0:
                    out.flush()
                print(f'{i+1}/{len(states)} {s["kind"]:5s} '
                      f'f={s["freq"]:4.1f} cpp  angle={s["angle"]:5.1f} deg  '
                      f'phase={s["phase"]:5.0f} deg  '
                      f'RMS={table["ACT_RMS"][i]:.4f}', flush=True)
            completed = True
        finally:
            # Do not overwrite an externally modified channel during cleanup.
            if np.array_equal(probe.get_data(), last_command):
                before_restore = master.get_counter()
                probe.set_data(initial)
                master.post_sems(DM_UPDATE_SEMID)
                deadline = time.monotonic() + args.timeout
                while master.get_counter() == before_restore and time.monotonic() < deadline:
                    time.sleep(0.002)
                restored = (master.get_counter() != before_restore and
                            np.array_equal(probe.get_data(), initial))
            out[0].header['RESTORED'] = restored
            out[0].header['COMPLETE'] = completed and restored
            out.flush()
            print(f'FITS: {args.output} (complete={completed}, restored={restored})', flush=True)
            if not restored:
                print('WARNING: DM channel NOT restored (ownership changed). Inspect hardware.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--beam', type=int, required=True)
    p.add_argument('--channel', type=int, default=3)
    p.add_argument('--shm-dir', type=Path, default=Path('/dev/shm'))
    p.add_argument('--output', type=Path, default=Path('baldr_spatial_transfer.fits'))
    p.add_argument('--angles', type=parse_range, default=parse_range('0:95:45'),
                   help='Degrees: start:stop:step (stop exclusive) or comma list')
    p.add_argument('--frequencies', type=parse_range, default=parse_range('0.5:5.5:0.5'),
                   help='Cycles/pupil: start:stop:step or comma list')
    p.add_argument('--phases', type=float, nargs='+', default=[0., 180., 90., 270.])
    p.add_argument('--rms', type=float, nargs='+', default=[0.03],
                   help='One or more RMS DM amplitudes over the 10-pitch pupil')
    p.add_argument('--frames', type=int, default=20, help='Fresh camera frames/state')
    p.add_argument('--camera-semid', type=int, default=2,
                   help='Dedicated producer-posted camera semaphore (default 2; RTC uses 1)')
    p.add_argument('--cycles', type=int, default=1)
    p.add_argument('--reference-every', type=int, default=40,
                   help='Insert zero-probe reference after every N probe states (0 disables)')
    p.add_argument('--settle-s', type=float, default=0.1)
    p.add_argument('--discard', type=int, default=3, help='Fresh camera frames discarded after settling')
    p.add_argument('--timeout', type=float, default=15.0)
    p.add_argument('--headroom-margin', type=float, default=0.005)
    p.add_argument('--checkpoint-every', type=int, default=10)
    p.add_argument('--dont_execute', action='store_false', help='dont send commands to the DM')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.beam < 1 or args.channel < 1 or args.channel > 99:
        p.error('Beam must be >=1 and DM channel must be 1..99 (channel 0 is the flat)')
    if args.camera_semid < 0 or args.camera_semid == 1:
        p.error('Camera semaphore must be >=0 and must not be 1 (reserved for RTC)')
    if args.frames < 1 or args.cycles < 1 or args.discard < 0 or args.reference_every < 0:
        p.error('Invalid count (frames/cycles >=1; discard/reference-every >=0)')
    if args.checkpoint_every < 1 or args.timeout <= 0 or args.settle_s < 0:
        p.error('Invalid timing/checkpoint option')
    if any(r <= 0 for r in args.rms) or any(f <= 0 for f in args.frequencies):
        p.error('All RMS amplitudes and spatial frequencies must be positive')
    if not 0 <= args.headroom_margin < 0.5:
        p.error('--headroom-margin must lie in [0, 0.5)')

    states = scan_states(args)
    x, y, active, pupil = geometry()
    nulls = sum(s['kind'] == 'PROBE' and
                make_probe(s['freq'], s['angle'], s['phase'], s['rms'],
                           x, y, active, pupil) is None for s in states)
    print(f'Beam {args.beam}, DM channel {args.channel}, camera semaphore {args.camera_semid}; '
          f'{sum(s["kind"] == "PROBE" for s in states)} probe states, '
          f'{sum(s["kind"] == "REF" for s in states)} references, '
          f'{args.frames} frames/state; {nulls} predicted DM sampling nulls.')
    if not args.dont_execute:
        print('PLAN ONLY. No shared memory opened. Add --execute to acquire.')
        return
    acquire(args, states)


if __name__ == '__main__':
    main()
