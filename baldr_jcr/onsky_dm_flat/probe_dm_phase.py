#!/usr/bin/env python3
"""Probe the DM, fit a non-linear model for common pupil phase, and propose a smooth flat offset. Intended for use onsky in open loop.

    python probe_dm_phase.py run --beam 1 --amplitude .03 --output dm_offset_b1.fits --execute
    python probe_dm_phase.py fit probes.fits --output fit.fits
    python probe_dm_phase.py analyse fit.fits

One iteration by default. Each flat update needs terminal 'y'. The main output
contains the accumulated approved offset; numbered files retain each step.
Keep the RTC/other DM writers held while probing. No RTC commands are sent.
DM convention: subchannel.set_data(command), then combined.post_sems(1).
Camera convention: xaosim get_counter/get_data, or a dedicated semaphore below.
No BaldrApp/config imports. Only an explicit DCS basis needs modal_basis.py.
The analytic fit is a monochromatic approximation, not full Fresnel propagation.
BCB
"""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import tempfile

import numpy as np
from astropy.io import fits

DEFAULT_DCS=Path('/home/bbarrer/Documents/baldr/dcs/baldr_jcr')
# Material-independent mask definitions: wavelength [m], phase shift [rad],
# DIAMETER [lambda/D]. Physical diameters converted with F#=21.2.
MASKS = {
    name: dict(wavelength=wave*1e-6, theta=np.pi/2, diameter_lamD=diam/(21.2*wave))
    for band, wave, diameters in [('J',1.20,[32,36,44,54,65]), ('H',1.65,[31,37,44,53,68])]
    for name, diam in zip([f'{band}{i}' for i in range(1,6)], diameters)
}

# Optical stop DIAMETER, not radius: 8 lambda/D passes radial frequencies
# through 4 cycles per illuminated pupil in the linear field approximation.
COLDSTOP_DIAM_LAMD = 8.0
MIXED_PUPIL_PITCHES = 10.0  # nominal illuminated DM diameter for probe design only
MIXED_LABELS = ['defocus','astigmatism_0','astigmatism_45'] + [
    f'{axis}_{quadrature}_{frequency}cpp_nominal'
    for frequency in (1,2) for axis in ('x','y') for quadrature in ('sin','cos')]

# Standard controls stay on the CLI; edit these installation/fit settings here.
SIMULATION = False  # --simulator selects the local MDS; default is on-sky.
MDS_HOST_ONSKY = 'mimir'
MDS_HOST_SIMULATOR = '127.0.0.1'
ACQUISITION_DEFAULTS = dict(
    channel=3, dcs_dir=DEFAULT_DCS, shm_dir=Path('/dev/shm'),
    coupling=.7, opd_per_cmd=3e-6,  # Gaussian width [pitches], OPD [m/DM unit]
    registration_pairs=10, registration_frames=1, corner_offset=3,
    settle_frames=3, registration_settle_frames=1,  # discard fresh frames after write
    settle_s=.5, registration_settle_s=.5,  # seconds; reduce for fast on-sky hardware
    timeout=120., mask_offset_um=200., mask_settle_s=1.,
    overwrite=False, save_intermediates=True, save_plots=True)
FIT_DEFAULTS = dict(
    background=None, pupil_threshold=.15, pupil_geometry=None,  # auto from clear image
    noise_floor=1., centroid_radius=3,  # ADU uncertainty floor; corner window [pixels]
    registration_prior_px=.2, registration_bound_px=1.,
    phase_bound=np.pi/2, max_nfev=200,  # near-flat phase branch [rad]
    dm_regularization=.02, dm_max_command=.1)  # regularisation; final offset limit
DM_NEIGHBOUR_REGULARIZATION = .15  # penalise horizontal/vertical differences; larger = smoother
DM_SMOOTH_SIGMA_PITCHES = 1.0  # Gaussian smoothing width [actuator pitches]; 0 disables taper

CORRECTION_GAIN = 1 #0.3  # fraction of each independently estimated offset to apply
MAX_CORRECTION_ITERATIONS = 1  # each application still requires terminal 'y'
REUSE_ITERATION_CALIBRATION = True  # same run only; assumes pupil, mask and illumination stay stable
UNCERTAINTY_BLOCKS = 3  # contiguous frame blocks account for short-timescale correlation

BAD_PIXEL_SIGMA = 10.  # isolated deviation from local median, in robust spatial sigmas
BAD_PIXEL_CONTRAST = 3.  # also exceed this multiple of the image 1st–99th percentile range
BAD_PIXEL_MASK = None  # optional FITS mask: nonzero = bad, same shape as camera subframe
REGISTRATION_ANGLE_STEP = 45  # degrees between modal registration starting orientations
REGISTRATION_SEARCH_NFEV = 60
REGISTRATION_SCALE_FRACTION = .4  # affine-coefficient search half-width / nominal pitch
REGISTRATION_CENTRE_FRACTION = .35  # centre search half-width / pupil radius
ROBUST_LOSS_SCALE = 3.  # soft-L1 transition in measurement standard errors

# Camera semaphore 1 belongs to the C++ RTC, including while it is open-loop.
# None uses fresh-counter reads without consuming any semaphore. Set an unused,
# producer-posted camera semaphore index only if it is reserved for this script.
CAMERA_SEMID = None
DM_UPDATE_SEMID = 1

def json_hdu(name,value):
    raw=json.dumps(value,allow_nan=False)
    return fits.BinTableHDU.from_columns([fits.Column(name='JSON',format=f'{len(raw)}A',array=[raw])],name=name)


def read_json(h,name):
    return json.loads(h[name].data['JSON'][0])


def basis_maps(directory,name,indices):
    if name=='Mixed':
        # These local analytic patterns need no DCS/AOtools imports. Their
        # frequencies refer to a nominal 10-pitch pupil, not an assumed camera
        # registration. The actual commands are saved and modelled in the fit.
        y,x=np.indices((12,12))
        x=(x-5.5)/(MIXED_PUPIL_PITCHES/2)
        y=(y-5.5)/(MIXED_PUPIL_PITCHES/2)
        patterns=[2*(x*x+y*y)-1,x*x-y*y,2*x*y]
        patterns += [fn(np.pi*frequency*coordinate) for frequency in (1,2)
                     for coordinate in (x,y) for fn in (np.sin,np.cos)]
        active=np.ones((12,12),bool)
        active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
        support=active & (x*x+y*y<=1)
        modes=np.asarray(patterns)[indices].copy()
        for mode in modes:
            mode-=np.mean(mode[support])
            mode[~active]=0
            mode/=np.max(np.abs(mode[active]))
        # Unit PEAK command: amplitude is bounded actuator excursion for Mixed.
        # Explicit DCS bases retain their original scaling below.
        return modes
    path=Path(directory)/'modal_basis.py'
    sys.path.insert(0,str(Path(directory).resolve()))
    try:
        spec=importlib.util.spec_from_file_location('_probe_dcs_basis',path)
        module=importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as exc:
            raise RuntimeError(f'DCS basis dependency missing: {exc.name}. Install it in the environment running this script (modal_basis.py requires aotools).') from exc
        basis=getattr(module,name)()
        modes=basis.modes_on_unit_disk(12,max(indices)+1)[:,indices].T.reshape(-1,12,12)
    finally:
        sys.path.pop(0)
    modes=np.asarray(modes,dtype=float)
    modes[:,0,0]=modes[:,0,-1]=modes[:,-1,0]=modes[:,-1,-1]=0
    if not np.isfinite(modes).all():
        raise ValueError('Non-finite basis')
    # Preserve DCS scaling (the norm argument in its implementation is unused).
    return modes


def rpc(address,command,timeout=10000):
    import zmq
    ctx=zmq.Context()
    sock=ctx.socket(zmq.REQ)
    try:
        sock.setsockopt(zmq.LINGER,0)
        sock.setsockopt(zmq.RCVTIMEO,timeout)
        sock.setsockopt(zmq.SNDTIMEO,timeout)
        sock.connect(address)
        sock.send_string(command)
        reply=sock.recv_string().strip()
        if reply.upper().startswith(('ERR','NACK','ERROR')):
            raise RuntimeError(reply)
        return reply
    finally:
        sock.close()
        ctx.term()


# Read and write using the same xaosim API as the DCS backend.
def camera_frames(camera, master, expected, n, settle, timeout, settle_s=0):
    """Wait for the DM, allow settling, then collect N distinct camera frames."""
    deadline=time.monotonic()+timeout
    tolerance=8*np.finfo(np.dtype(master.npdtype)).eps
    while not np.allclose(master.get_data(),expected,rtol=0,atol=tolerance):
        if time.monotonic()>deadline:
            raise TimeoutError('Combined DM did not reach the probe')
        time.sleep(.002)
    time.sleep(settle_s)
    last=camera.get_counter()
    if CAMERA_SEMID is not None:
        camera.catch_up_with_sem(CAMERA_SEMID)
    images=[]
    counters=[]
    timestamps=[]
    while len(images)<n:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise TimeoutError(f'Camera stalled: {len(images)}/{n} frames')
        if CAMERA_SEMID is not None:
            import posix_ipc
            try:
                camera.sems[CAMERA_SEMID].acquire(timeout=remaining)
            except posix_ipc.BusyError:
                raise TimeoutError('Camera semaphore timed out') from None
        # A counter check is still needed: stale semaphore posts are not frames.
        counter=camera.get_counter()
        if counter<last:
            raise RuntimeError('Camera counter reset during acquisition')
        if counter==last:
            time.sleep(.001)
            continue
        if camera.mtdata['naxis']==3:
            image=camera.get_latest_data_slice(semid=None).copy()
        else:
            image=camera.get_data().copy()
        if camera.get_counter()!=counter:
            continue  # producer changed image while copying
        if not np.allclose(master.get_data(),expected,rtol=0,atol=tolerance):
            raise RuntimeError('Combined DM changed; hold other DM writers before probing')
        last=counter
        if settle:
            settle-=1
            continue
        images.append(image.astype(float))
        counters.append(int(counter))
        timestamps.append(time.time_ns())
    return np.stack(images),counters,timestamps


def save_acquisition(path,settings,modes,baseline,combined,clear,shots,records,complete,flat=None):
    header=fits.Header()
    header['COMPLETE']=complete
    header['BEAM']=settings['beam']
    header['CHANNEL']=settings['channel']
    header['AMPL']=settings['amplitude']
    header['BASIS']=settings['basis']
    header['BUNIT']='DM command units'
    hdus=[fits.PrimaryHDU(header=header),
          json_hdu('SETTINGS',settings),json_hdu('RECORDS',records),fits.ImageHDU(modes,name='MODES'),
          fits.ImageHDU(baseline,name='CHANNEL_BEFORE'),fits.ImageHDU(combined,name='DM_BEFORE')]
    if flat is not None:
        hdus.append(fits.ImageHDU(flat,name='FLAT_BEFORE'))
    if clear is not None:
        hdus.append(fits.ImageHDU(clear,name='CLEAR_FRAMES'))
        hdus.append(fits.ImageHDU(clear.mean(axis=0),name='CLEAR_MEAN'))
    if shots:
        # Registration states may contain one frame while modal states contain N.
        # Trailing NaNs are padding, never measurements; preserve the true counts.
        counts=np.array([len(shot) for shot in shots])
        cube=np.full((len(shots),counts.max(),*shots[0].shape[1:]),np.nan)
        for i,shot in enumerate(shots):
            cube[i,:len(shot)]=shot
        hdus.append(fits.ImageHDU(cube,name='PROBE_FRAMES'))
        hdus.append(fits.ImageHDU(counts,name='FRAME_COUNTS'))
        hdus.append(fits.ImageHDU(np.array([shot.mean(axis=0) for shot in shots]),name='PROBE_MEANS'))
        pairs={}
        for image,record in zip(shots,records):
            if record['sign']:
                pairs.setdefault((record['kind'],record['cycle'],record['mode']),{})[record['sign']]=image.mean(axis=0)
        differences=[pair[1]-pair[-1] for pair in pairs.values() if 1 in pair and -1 in pair]
        if differences:
            hdus.append(fits.ImageHDU(np.stack(differences),name='DIFFERENCES'))
    temp=path.with_suffix('.fits.partial')
    with fits.HDUList(hdus) as h:
        h.writeto(temp,overwrite=True,checksum=True)
    os.replace(temp,path)


def acquire(args):
    if not args.execute:
        print(f'PLAN ONLY: beam {args.beam}, channel {args.channel}, {args.basis} modes {args.modes}, ±{args.amplitude} DM units, {args.frames} frames/sign, {args.cycles} cycles; four inner-corner ±{args.registration_amplitude} registration pokes ({args.registration_pairs} pairs, {args.registration_frames} frames/sign); mask {args.mask}.\nUse --execute for acquisition; keep the RTC off. No hardware commands sent.')
        return
    cache=getattr(args,'calibration_cache',None)
    modes=basis_maps(args.dcs_dir,args.basis,args.modes)
    corners=[(args.corner_offset,args.corner_offset),(args.corner_offset,11-args.corner_offset),
             (11-args.corner_offset,args.corner_offset),(11-args.corner_offset,11-args.corner_offset)]
    corner_modes=np.zeros((4,12,12))
    for k,(row,col) in enumerate(corners):
        corner_modes[k,row,col]=1
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'{args.output} exists; use --overwrite')
    if args.output.with_suffix('.fits.partial').exists() and not args.overwrite:
        raise FileExistsError('Partial output already exists; use --overwrite')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    settings=dict(beam=args.beam,channel=args.channel,amplitude=args.amplitude,basis=args.basis,
                  mode_indices=args.modes,frames=args.frames,cycles=args.cycles,settle_frames=args.settle_frames,
                  mask=args.mask,mask_parameters=MASKS[args.mask],mask_offset_um=args.mask_offset_um,
                  coupling=args.coupling,opd_per_cmd=args.opd_per_cmd,corner_rows_cols=corners,
                  registration_amplitude=args.registration_amplitude,registration_pairs=args.registration_pairs,
                  registration_frames=args.registration_frames,registration_settle_s=args.registration_settle_s,
                  registration_settle_frames=args.registration_settle_frames,settle_s=args.settle_s,
                  clear_record=None,restored=False,format_version=3)
    if cache is not None:
        settings['registration_cache']=cache['registration']
        settings['calibration_source']=cache['source']
    if args.basis=='Mixed':
        settings['mode_labels']=[MIXED_LABELS[i] for i in args.modes]
        settings['mode_normalisation']='unit peak actuator command; pupil mean removed'
        settings['probe_nominal_pupil_pitches']=MIXED_PUPIL_PITCHES
    from xaosim.shmlib import shm
    with ExitStack() as stack:
        paths=dict(probe=args.shm_dir/f'dm{args.beam}disp{args.channel:02d}.im.shm',
                   master=args.shm_dir/f'dm{args.beam}.im.shm',
                   camera=args.shm_dir/f'baldr{args.beam}.im.shm',
                   flat=args.shm_dir/f'dm{args.beam}disp00.im.shm')
        streams={}
        identities={}
        for name,path in paths.items():
            stat=path.stat()  # all streams must already exist
            identities[name]=(stat.st_dev,stat.st_ino)
            stream=shm(str(path),nosem=not(name=='master' or (name=='camera' and CAMERA_SEMID is not None)))
            stack.callback(stream.close,erase_file=False)
            for sem in getattr(stream,'sems',[]):
                stack.callback(sem.close)
            opened=os.fstat(stream.fd)
            if (opened.st_dev,opened.st_ino)!=identities[name]:
                raise RuntimeError('SHM replaced while opening')
            streams[name]=stream
        source=streams['probe']
        master=streams['master']
        camera=streams['camera']
        baseline=source.get_data().copy()
        combined=master.get_data().copy()
        flat=streams['flat'].get_data().copy()
        settings['shm_identity']={name:identities[name] for name in ('flat','master','probe')}
        settings['camera_identity']=identities['camera']
        if cache is not None:
            if any(tuple(cache['shm_identity'][name])!=identities[name] for name in ('flat','master','probe')) or tuple(cache['camera_identity'])!=identities['camera']:
                raise RuntimeError('Streams replaced between passes; start again to recalibrate')
        expected_flat=getattr(args,'expected_flat',None)
        if expected_flat is not None and not np.array_equal(flat,expected_flat):
            raise RuntimeError('Flat changed between passes')
        settings['shm_dir']=str(args.shm_dir.resolve())
        if baseline.shape!=(12,12) or flat.shape!=(12,12) or baseline.dtype.kind!='f':
            raise ValueError('Expected float 12x12 DM channel')
        active=np.ones((12,12),bool)
        active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
        # Preserve +/- symmetry when a requested probe exceeds actuator headroom.
        # Store the actual channel commands below; the phase fit uses those commands.
        headroom=np.maximum(0,np.minimum(combined,1-combined))
        headroom[~active]=0
        limited=sum(np.any(np.abs(amplitude*mode)>headroom+1e-12)
                    for mode,amplitude in [(m,args.amplitude) for m in modes]+
                                          [(m,args.registration_amplitude) for m in corner_modes])
        settings['headroom_limited_patterns']=int(limited)
        if limited:
            print(f'WARNING: {limited} probe patterns limited symmetrically to available DM headroom; actual commands will be saved.')
        last_written=baseline.copy()
        mask_out=False
        mask_uncertain=False
        clear=None
        shots=[]
        records=[]
        complete=False
        def write(command):
            nonlocal last_written
            for name,path in paths.items():
                stat=path.stat()
                if (stat.st_dev,stat.st_ino)!=identities[name]:
                    raise RuntimeError(f'{name} SHM replaced')
            if not np.array_equal(source.get_data(),last_written):
                raise RuntimeError('Probe channel modified by another writer')
            last_written=np.asarray(command,dtype=baseline.dtype)
            source.set_data(last_written)
            master.post_sems(DM_UPDATE_SEMID)
        def on_signal(signum,frame):
            raise KeyboardInterrupt(f'Signal {signum}')
        previous=signal.signal(signal.SIGTERM,on_signal)
        stack.callback(signal.signal,signal.SIGTERM,previous)
        try:
            save_acquisition(args.output,settings,modes,baseline,combined,clear,shots,records,False,flat)
            if cache is not None:
                clear=np.asarray(cache['clear_frames']).copy()
                settings['clear_record']=cache['clear_record']
                print('Reusing initial clear pupil and corner registration; acquiring fresh modal probes.',flush=True)
            else:
                mask_uncertain=True
                rpc(args.mds,f'moverel BMY{args.beam} {args.mask_offset_um}')
                mask_out=True
                mask_uncertain=False
                time.sleep(args.mask_settle_s)
                clear,counts,times=camera_frames(camera,master,combined,args.frames,args.settle_frames,args.timeout,args.settle_s)
                settings['clear_record']=dict(counters=counts,time_ns=times)
                mask_uncertain=True
                rpc(args.mds,f'moverel BMY{args.beam} {-args.mask_offset_um}')
                mask_out=False
                mask_uncertain=False
                time.sleep(args.mask_settle_s)
            orders=[('baseline',-1,0,0)]
            # Short neighbouring +/- states; reverse order on alternating pairs
            # to reduce a systematic association between probe sign and time drift.
            orders += [('registration',pair,k,sign) for k in range(4)
                       for pair in range(args.registration_pairs if cache is None else 0)
                       for sign in ((1,-1) if pair%2==0 else (-1,1))]
            for cycle in range(args.cycles):
                for k in range(len(modes)):
                    for sign in ((1,-1) if cycle%2==0 else (-1,1)):
                        orders.append(('modal',cycle,k,sign))
            for kind,cycle,k,sign in orders:
                amplitude=args.registration_amplitude if kind=='registration' else args.amplitude
                mode=corner_modes[k] if kind=='registration' else modes[k]
                delta=sign*np.clip(amplitude*mode,-headroom,headroom)
                write(baseline+delta)
                is_registration=kind=='registration'
                nframes=args.registration_frames if is_registration else args.frames
                settle=args.registration_settle_frames if is_registration else args.settle_frames
                settle_s=args.registration_settle_s if is_registration else args.settle_s
                command_time=time.time_ns()
                frames,counts,times=camera_frames(camera,master,combined+(last_written.astype(float)-baseline),nframes,settle,args.timeout,settle_s)
                shots.append(frames)
                records.append(dict(kind=kind,cycle=cycle,mode=k,
                                    mode_index=args.modes[k] if kind=='modal' else None,sign=sign,
                                    requested_amplitude=amplitude if sign else 0,
                                    applied_peak_delta=float(np.max(np.abs(last_written.astype(float)-baseline))),
                                    command_time_ns=command_time,counters=counts,time_ns=times,
                                    command=last_written.tolist()))
                # No FITS writes or console output between the two signs of a
                # registration pair. Checkpoint only after all pairs for a corner.
                corner_done=is_registration and cycle==args.registration_pairs-1 and sign==( -1 if cycle%2==0 else 1)
                if not is_registration or corner_done:
                    save_acquisition(args.output,settings,modes,baseline,combined,clear,shots,records,False,flat)
                    print(f'Saved {kind}, index {k}'+(f', sign {sign:+d}, cycle {cycle+1}' if kind=='modal' else ''),flush=True)
            complete=True
        finally:
            failures=[]
            try:
                write(baseline)
            except Exception as exc:
                failures.append(f'DM restoration failed: {exc}')
            if mask_uncertain:
                failures.append('Mask command outcome uncertain; inspect mask position before moving it again')
            elif mask_out:
                try:
                    rpc(args.mds,f'moverel BMY{args.beam} {-args.mask_offset_um}')
                except Exception as exc:
                    failures.append(f'Mask restoration failed: {exc}')
            settings['restored']=not failures
            settings['restoration_errors']=failures
            save_acquisition(args.output,settings,modes,baseline,combined,clear,shots,records,complete and not failures,flat)
            if failures:
                raise RuntimeError('; '.join(failures))
    print(f'Acquisition saved: {args.output}. Original probe channel and mask restored. Flat and RTC untouched.')


# Fit registration/phase, then solve for the actual allowed DM command.
def constrained_command_fit(H, target, gradient, affine, diameter, opd_ptt, low, high, regularization):
    """Optimise the actual allowed command: band-limited, zero pupil PTT, bounded.

    Gaussian smoothing is a spectral penalty inside this solve, not post-filtering.
    """
    from scipy.linalg import null_space
    from scipy.optimize import minimize, LinearConstraint
    active=np.ones((12,12),bool)
    active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
    fx,fy=np.meshgrid(np.fft.fftfreq(12),np.fft.fftfreq(12))
    freq=np.linalg.solve(affine[:,:2].T,np.vstack([fx.ravel(),fy.ravel()]))
    radial=np.hypot(*freq).reshape(12,12)*diameter
    band=radial<=COLDSTOP_DIAM_LAMD/2+1e-12
    reverse=(-np.arange(12))%12
    band &= band[np.ix_(reverse,reverse)]
    impulses=np.eye(144).reshape(144,12,12)
    projector=np.fft.ifft2(np.fft.fft2(impulses)*band).real.reshape(144,144)
    values,vectors=np.linalg.eigh(projector)
    fourier_basis=vectors[:,values>.5]
    constraints=np.eye(144)[~active.ravel()]
    rows=np.zeros((3,144))
    rows[:,active.ravel()]=opd_ptt
    rows/=np.maximum(np.linalg.norm(rows,axis=1,keepdims=True),1e-30)
    constraints=np.vstack([constraints,rows])
    if np.any(low>0) or np.any(high<0):
        raise ValueError('DM baseline outside available headroom')
    fixed=high-low<1e-12
    if np.any(fixed):
        constraints=np.vstack([constraints,np.eye(144)[np.flatnonzero(active.ravel())[fixed]]])
    Z=fourier_basis@null_space(constraints@fourier_basis,rcond=1e-10)
    if not Z.shape[1]:
        raise ValueError('No free band-limited, PTT-free correction commands')
    Za=Z[active.ravel()]
    # Inverse Gaussian weighting suppresses high frequencies without changing
    # the optimised command afterward. Neighbour regularisation acts on Za too.
    taper=np.exp(-2*np.pi**2*DM_SMOOTH_SIGMA_PITCHES**2*(fx*fx+fy*fy))
    spectrum=np.fft.fft2(Z.T.reshape(-1,12,12),norm='ortho').reshape(Z.shape[1],144).T
    spectral_penalty=np.sqrt(np.maximum(1/taper.ravel()**2-1,0))[:,None]*spectrum
    scale=max(float(np.linalg.norm(H@Za,ord=2)),1e-12)
    alpha=regularization*scale
    beta=DM_NEIGHBOUR_REGULARIZATION*scale
    design=np.vstack([H@Za,alpha*Za,beta*gradient@Za,
                      alpha*spectral_penalty.real,alpha*spectral_penalty.imag])
    rhs=np.r_[target,np.zeros(len(design)-len(target))]
    normalizer=max(float(np.linalg.norm(design,ord=2))**2,1e-20)
    Q=design.T@design/normalizer
    b=design.T@rhs/normalizer
    result=minimize(lambda z:.5*z@Q@z-b@z,np.zeros(Z.shape[1]),jac=lambda z:Q@z-b,
                    constraints=[LinearConstraint(Za[~fixed],low[~fixed],high[~fixed])],
                    method='SLSQP',options=dict(maxiter=500,ftol=1e-12))
    full=(Z@result.x).reshape(12,12)
    full[~active]=0
    full.ravel()[np.flatnonzero(active.ravel())[fixed]]=0  # numerical roundoff only
    if np.max(np.abs(constraints@full.ravel()))>1e-9:
        raise ValueError('Command constraints not satisfied')
    # Only remove floating-point bound overshoot, never reshape the solution.
    vector=full[active]
    ratios=[1.]
    if np.any(vector>0):
        ratios.append(float(np.min(high[vector>0]/vector[vector>0])))
    if np.any(vector<0):
        ratios.append(float(np.min(low[vector<0]/vector[vector<0])))
    bound_scale=max(0.,min(ratios))
    if bound_scale<1-1e-7:
        raise ValueError('Constrained command solver exceeded actuator bounds')
    if bound_scale<1:
        full*=bound_scale*(1-1e-12)
    return full,band,radial,result


def fit_file(args):
    from scipy.ndimage import median_filter, gaussian_filter, label
    from scipy.interpolate import RectBivariateSpline
    from scipy.optimize import curve_fit, least_squares
    from scipy.special import j0, j1
    from scipy.sparse import lil_matrix

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f'{args.output} exists; use --overwrite')
    if args.input.resolve()==args.output.resolve():
        raise ValueError('Fit output must differ from input acquisition')
    with fits.open(args.input) as h:
        if not h[0].header.get('COMPLETE',False):
            raise ValueError('Acquisition incomplete; inspect restoration status')
        settings=read_json(h,'SETTINGS')
        records=read_json(h,'RECORDS')
        if settings.get('format_version') not in (2,3):
            raise ValueError('This fit requires a new acquisition with corner-registration measurements (format_version 2 or 3). Older Fresnel fit files remain readable by analyse_probe_phase.py.')
        clear_measured=h['CLEAR_MEAN'].data.astype(float)
        clear=clear_measured.copy()
        frames=h['PROBE_FRAMES'].data.astype(float)
        before=h['CHANNEL_BEFORE'].data.astype(float)
        combined=h['DM_BEFORE'].data.astype(float)
        flat=h['FLAT_BEFORE'].data.astype(float) if 'FLAT_BEFORE' in h else None
        frame_counts=h['FRAME_COUNTS'].data.astype(int) if 'FRAME_COUNTS' in h else np.full(len(frames),frames.shape[1])
    mask=MASKS[settings['mask']]
    wavelength=mask['wavelength']
    theta=mask['theta']
    mu=float(np.angle(np.exp(1j*theta)-1))
    M=abs(np.exp(1j*theta)-1)
    yy,xx=np.indices(clear.shape)
    border=np.zeros(clear.shape,bool)
    border[[0,-1],:]=True
    border[:,[0,-1]]=True
    background=float(np.median(clear[border])) if args.background is None else args.background
    clear=np.maximum(clear-background,0)
    means=np.array([shot[:count].mean(axis=0) for shot,count in zip(frames,frame_counts)])
    invalid=(~np.isfinite(clear)) | ~np.isfinite(means).all(axis=0)
    clear=np.where(np.isfinite(clear),clear,0)
    # Detect isolated defects without smoothing the intensities used by the fit.
    smooth=median_filter(clear,size=3)
    deviation=clear-smooth
    scatter=max(1.4826*float(np.median(np.abs(deviation-np.median(deviation)))),args.noise_floor)
    threshold=max(BAD_PIXEL_SIGMA*scatter,BAD_PIXEL_CONTRAST*np.ptp(np.percentile(clear,[1,99])))
    bad_pixels=(~np.isfinite(clear)) | (np.abs(deviation)>threshold)
    # A high-contrast pupil edge is not by itself a hot pixel: require isolation.
    local_range=gaussian_filter(np.abs(deviation),1)
    bad_pixels &= (~np.isfinite(clear)) | (np.abs(deviation)>4*local_range)
    # Also catch defects appearing only during probing; require an isolated spike.
    for image in means:
        clean=np.where(np.isfinite(image),image,0)
        dev=clean-median_filter(clean,3)
        noise=max(1.4826*float(np.median(np.abs(dev-np.median(dev)))),args.noise_floor)
        threshold=max(BAD_PIXEL_SIGMA*noise,BAD_PIXEL_CONTRAST*np.ptp(np.percentile(clean,[1,99])))
        bad_pixels |= (np.abs(dev)>threshold) & (np.abs(dev)>4*gaussian_filter(np.abs(dev),1))
    bad_pixels |= invalid
    if BAD_PIXEL_MASK is not None:
        supplied=fits.getdata(BAD_PIXEL_MASK)
        if supplied.shape!=clear.shape:
            raise ValueError('Bad-pixel mask shape differs from camera subframe')
        bad_pixels |= supplied!=0
    smooth=np.where(np.isfinite(smooth),smooth,0)
    regions,count=label(smooth>args.pupil_threshold*np.percentile(smooth,99))
    if not count:
        raise ValueError('No clear pupil found')
    sizes=np.bincount(regions.ravel())
    sizes[0]=0
    pupil=(regions==sizes.argmax()) & ~bad_pixels
    if pupil.sum()<12:
        raise ValueError('Too few illuminated pixels; check background/pupil threshold')
    # Geometry of the optical pupil is distinct from the fitted DM centre.
    cx=float((xx[pupil].min()+xx[pupil].max())/2)
    cy=float((yy[pupil].min()+yy[pupil].max())/2)
    radius=float(np.max(np.hypot(xx[pupil]-cx,yy[pupil]-cy)))
    if args.pupil_geometry is not None:
        cx,cy,radius=args.pupil_geometry
    if radius<=0:
        raise ValueError('Pupil radius must be positive')
    pupil &= np.hypot(xx-cx,yy-cy)<=radius+1e-6
    if pupil.sum()<12:
        raise ValueError('Pupil geometry excludes the illuminated region')
    norm=float(np.median(clear[pupil]))
    A=clear/norm
    # Spline interpolation followed by
    # a local elliptical Gaussian fit to each absolute +/- intensity difference.
    def gaussian(xy,amp,x0,y0,sx,sy,angle,offset):
        x,y=xy
        u=(x-x0)*np.cos(angle)+(y-y0)*np.sin(angle)
        v=-(x-x0)*np.sin(angle)+(y-y0)*np.cos(angle)
        return offset+amp*np.exp(-.5*((u/sx)**2+(v/sy)**2))

    corner_images=[]
    centres=[]
    gaussian_parameters=[]
    corner_errors=[]
    pair_gaps=[]
    cached_registration=settings.get('registration_cache')
    for k in range(4):
        if cached_registration is not None:
            corner_images.append(np.asarray(cached_registration['images'][k]))
            corner_errors.append(np.asarray(cached_registration['errors'][k],float))
            centres.append(np.asarray(cached_registration['centres'][k]))
            gaussian_parameters.append(np.asarray(cached_registration['gaussians'][k]))
            continue
        pairs={}
        for i,r in enumerate(records):
            if r['kind']=='registration' and r['mode']==k:
                pairs.setdefault(r['cycle'],{})[r['sign']]=i
        differences=[]
        for pair in pairs.values():
            if set(pair)!={-1,1}:
                raise ValueError('Incomplete registration pair')
            differences.append(means[pair[1]]-means[pair[-1]])
            if 'time_ns' in records[pair[1]] and 'time_ns' in records[pair[-1]]:
                pair_gaps.append(abs(float(np.mean(records[pair[1]]['time_ns']))-float(np.mean(records[pair[-1]]['time_ns'])))/1e9)
        if not differences:
            raise ValueError(f'No registration pairs for corner {k}')
        delta=np.mean(differences,axis=0)
        corner_images.append(delta)
        corner_errors.append(np.std(differences,axis=0,ddof=1)/np.sqrt(len(differences)) if len(differences)>1 else np.full(clear.shape,np.nan))
        # Broad, persistent response wins over an isolated bright detector pixel.
        delta_clean=np.where(np.isfinite(delta),delta,0)
        image=np.abs(np.where(bad_pixels,median_filter(delta_clean,3),delta_clean))
        image=gaussian_filter(image,.65)
        py,px=np.unravel_index(np.argmax(np.where(pupil,image,-np.inf)),image.shape)
        x=np.arange(max(0,px-args.centroid_radius),min(image.shape[1],px+args.centroid_radius+1))
        y=np.arange(max(0,py-args.centroid_radius),min(image.shape[0],py+args.centroid_radius+1))
        if min(len(x),len(y))<4:
            raise ValueError('Registration response too close to image boundary')
        fine_x=np.linspace(x[0],x[-1],len(x)*5)
        fine_y=np.linspace(y[0],y[-1],len(y)*5)
        fx,fy=np.meshgrid(fine_x,fine_y)
        data=RectBivariateSpline(y,x,image[np.ix_(y,x)])(fine_y,fine_x)
        amplitude=max(float(data.max()-np.median(data)),1e-9)
        try:
            popt,_=curve_fit(gaussian,(fx.ravel(),fy.ravel()),data.ravel(),
                            p0=[amplitude,px,py,1,1,0,float(np.median(data))],
                            bounds=([0,x[0],y[0],.2,.2,-np.pi,-np.inf],
                                    [np.inf,x[-1],y[-1],len(x),len(y),np.pi,np.inf]),maxfev=10000)
            if np.linalg.norm(popt[1:3]-[px,py])>args.centroid_radius:
                raise ValueError('Displaced Gaussian centre')
        except (RuntimeError,ValueError):
            print(f'WARNING: corner {k} Gaussian unreliable; modal probes will refine registration.')
            popt=np.array([amplitude,px,py,1,1,0,float(np.median(data))])
        centres.append(popt[1:3])
        gaussian_parameters.append(popt)
    centres=np.asarray(centres)
    corners=np.asarray(settings['corner_rows_cols'],float)[:,::-1]-5.5
    corner_design=np.column_stack([corners,np.ones(4)])
    # Known actuator order defines opposite corners (0,3) and (1,2), including flips.
    v=centres[3]-centres[0]
    w=centres[2]-centres[1]
    crossing=np.linalg.lstsq(np.column_stack([v,-w]),centres[1]-centres[0],rcond=None)[0]
    centre=centres[0]+crossing[0]*v
    affine=np.column_stack([np.linalg.lstsq(corners,centres-centre,rcond=None)[0].T,centre])
    corner_affine=affine.copy()
    if np.linalg.cond(affine[:,:2])>10:
        print('WARNING: degenerate corner registration; using pupil-centred modal starts.')
        affine=np.array([[2*radius/MIXED_PUPIL_PITCHES,0,cx],[0,2*radius/MIXED_PUPIL_PITCHES,cy]])
    if cached_registration is not None:
        affine=np.asarray(cached_registration['affine'])
    print('Initial DM-to-camera affine matrix:\n',affine,flush=True)

    # Uniform circular-pupil reference amplitude, in units of incident amplitude:
    # b(rho) = integral_0^(pi*d/2) J1(t) J0(rho*t) dt, d = mask DIAMETER in lambda/D.
    # Gauss-Legendre quadrature avoids a propagation/config dependency. Obscurations,
    # scintillation and broadband changes of b are intentionally not modelled.
    nodes,weights=np.polynomial.legendre.leggauss(128)
    cutoff=np.pi*mask['diameter_lamD']/2
    t=(nodes+1)*cutoff/2
    rho=np.hypot(xx-cx,yy-cy)/radius
    B=np.sum(j0(rho[...,None]*t)*j1(t)*weights,axis=-1)*cutoff/2
    selected=[i for i,r in enumerate(records) if r['kind']!='registration']
    observed=(means[selected]-background)/norm
    # Per-block mean errors plus between-cycle atmospheric scatter. Repeated
    # exposures remain separate observations, so do not divide cycle scatter by sqrt(cycles).
    within=[]
    for i in selected:
        shot=frames[i,:frame_counts[i]]
        error=shot.std(axis=0,ddof=1)/np.sqrt(len(shot)) if len(shot)>1 else np.zeros(clear.shape)
        blocks=np.array([block.mean(axis=0) for block in np.array_split(shot,min(UNCERTAINTY_BLOCKS,len(shot)))])
        if len(blocks)>1:
            error=np.maximum(error,blocks.std(axis=0,ddof=1)/np.sqrt(len(blocks)))
        within.append(np.maximum(error,args.noise_floor)/norm)
    within=np.asarray(within)
    atmosphere=np.zeros_like(within)
    repeated_groups=0
    groups={}
    for j,i in enumerate(selected):
        record=records[i]
        if record['kind']=='modal':
            groups.setdefault((record['mode'],record['sign']),[]).append(j)
    floors=[]
    for group in groups.values():
        if len(group)<2:
            continue
        scatter=np.std(observed[group],axis=0,ddof=1)
        atmosphere[group]=scatter
        floors.append(scatter)
        repeated_groups+=1
    # Pair differences measure decorrelation between signs; common-mode drift
    # is already captured by the separate same-sign repeat scatter above.
    for mode in {key[0] for key in groups}:
        pairs={}
        for j,i in enumerate(selected):
            record=records[i]
            if record['kind']=='modal' and record['mode']==mode:
                pairs.setdefault(record['cycle'],{})[record['sign']]=j
        pairs=[v for v in pairs.values() if set(v)=={-1,1}]
        if len(pairs)>1:
            differences=np.array([observed[v[1]]-observed[v[-1]] for v in pairs])
            scatter=differences.std(axis=0,ddof=1)/np.sqrt(2)
            for pair in pairs:
                for j in pair.values():
                    atmosphere[j]=np.maximum(atmosphere[j],scatter)
    if floors:
        pooled=np.median(floors,axis=0)
        for j,i in enumerate(selected):
            if records[i]['kind']=='baseline':
                atmosphere[j]=pooled
    else:
        print('WARNING: no repeated modal cycles; atmospheric variability cannot be estimated.')
    sigma=np.maximum(within,atmosphere)
    commands=np.asarray([np.asarray(records[i]['command'])-before for i in selected])
    active=np.ones((12,12),bool)
    active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
    ay,ax=np.indices((12,12))
    actuators=np.column_stack([ax[active]-5.5,ay[active]-5.5])
    commands=commands[:,active]
    coupling=settings['coupling'] if args.coupling is None else args.coupling
    gain=settings['opd_per_cmd'] if args.opd_per_cmd is None else args.opd_per_cmd
    pixels=np.column_stack([xx[pupil],yy[pupil]])
    ap=A[pupil]
    bp=B[pupil]
    base=ap+M*M*bp*bp
    response=2*M*np.sqrt(ap)*bp
    n=pupil.sum()
    exposures=len(selected)
    obs=observed[:,pupil]
    err=sigma[:,pupil]

    def prediction(parameters):
        transform=parameters[:6].reshape(2,3)
        # Evaluate circular Gaussian influences in DM coordinates. The affine
        # transform then carries rotation, scale, shear and parity into the image.
        uv=np.linalg.solve(transform[:,:2],(pixels-transform[:,2]).T).T
        distance=np.sum((uv[:,None,:]-actuators[None,:,:])**2,axis=2)
        influences=np.exp(-distance/coupling**2)
        probe_phase=(2*np.pi*gain/wavelength)*(commands @ influences.T)
        angle=mu-parameters[6:][None,:]-probe_phase
        return base[None,:]+response[None,:]*np.cos(angle),probe_phase

    # Profile out one common pixel phase while fitting only six affine parameters.
    # Multiple pupil-centred orientations avoid locking onto a bad corner estimate.
    target=(obs-base)/err
    def modal_registration(vector, return_phase=False):
        _,probe=prediction(np.r_[vector,np.zeros(n)])
        c=response[None,:]*np.cos(mu-probe)/err
        q=response[None,:]*np.sin(mu-probe)/err
        cc=np.sum(c*c,axis=0)
        qq=np.sum(q*q,axis=0)
        cq=np.sum(c*q,axis=0)
        yc=np.sum(target*c,axis=0)
        yq=np.sum(target*q,axis=0)
        det=np.maximum(cc*qq-cq*cq,1e-20)
        phi=np.arctan2((yq*cc-yc*cq)/det,(yc*qq-yq*cq)/det)
        phi=np.clip(phi,-args.phase_bound*.99,args.phase_bound*.99)
        if return_phase:
            return phi
        model,_=prediction(np.r_[vector,phi])
        return ((model-obs)/err).ravel()

    pitch=2*radius/MIXED_PUPIL_PITCHES
    starts=[affine]
    for degrees in ([] if cached_registration is not None else range(0,360,REGISTRATION_ANGLE_STEP)):
        angle=np.deg2rad(degrees)
        rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        for parity in (-1,1):
            starts.append(np.column_stack([pitch*rotation@np.diag([1,parity]),[cx,cy]]))
    search_limits=np.tile([REGISTRATION_SCALE_FRACTION*pitch]*2+
                          [max(args.registration_bound_px,REGISTRATION_CENTRE_FRACTION*radius)],2)
    best=None
    for start in starts:
        if np.linalg.norm(start[:,2]-[cx,cy])>radius:
            continue
        if np.linalg.svd(start[:,:2],compute_uv=False).min()<.5*pitch:
            continue
        lo=start.ravel()-search_limits
        hi=start.ravel()+search_limits
        lo[[2,5]]=np.maximum(lo[[2,5]],np.array([cx,cy])-search_limits[[2,5]])
        hi[[2,5]]=np.minimum(hi[[2,5]],np.array([cx,cy])+search_limits[[2,5]])
        if np.any(lo>=hi):
            continue
        try:
            trial=least_squares(modal_registration,np.clip(start.ravel(),lo+1e-9,hi-1e-9),
                                bounds=(lo,hi),loss='soft_l1',f_scale=ROBUST_LOSS_SCALE,
                                max_nfev=REGISTRATION_SEARCH_NFEV)
        except np.linalg.LinAlgError:
            continue
        singular=np.linalg.svd(trial.x.reshape(2,3)[:,:2],compute_uv=False)
        if singular.min()<.5*pitch or singular.max()>1.8*pitch:
            continue
        if best is None or trial.cost<best.cost:
            best=trial
    if best is None:
        raise ValueError('No plausible modal registration; inspect probe and pupil images')
    affine=best.x.reshape(2,3)
    # Corners inconsistent with the modal result get a weaker prior, not a veto.
    corner_disagreement=np.linalg.norm(corner_design@affine.T-centres,axis=1)
    corner_prior=np.maximum(args.registration_prior_px,corner_disagreement)
    print('Modal DM-to-camera affine matrix:\n',affine,flush=True)

    def residual(parameters):
        model,_=prediction(parameters)
        # A pixel-space prior is independent of how affine coefficients are scaled.
        shift=corner_design @ parameters[:6].reshape(2,3).T-centres
        return np.concatenate([((model-obs)/err).ravel(),(shift/corner_prior[:,None]).ravel()])

    initial=np.r_[affine.ravel(),modal_registration(affine.ravel(),return_phase=True)]
    # Each image pixel depends only on its own phase and the six shared affine terms.
    pattern=lil_matrix((exposures*n+8,n+6),dtype=int)
    pattern[:,:6]=1
    for k in range(exposures):
        pattern[k*n+np.arange(n),6+np.arange(n)]=1
    scale=max(float(np.linalg.svd(affine[:,:2],compute_uv=False).min()),1e-3)
    limits=np.tile([.2*scale,.2*scale,args.registration_bound_px],2)
    lower=np.r_[affine.ravel()-limits,np.full(n,-args.phase_bound)]
    upper=np.r_[affine.ravel()+limits,np.full(n,args.phase_bound)]
    result=least_squares(residual,initial,bounds=(lower,upper),jac_sparsity=pattern.tocsr(),
                         x_scale='jac',loss='soft_l1',f_scale=ROBUST_LOSS_SCALE,max_nfev=args.max_nfev,ftol=1e-7,xtol=1e-7,gtol=1e-7)
    predicted,probes=prediction(result.x)
    phase=np.full(clear.shape,np.nan)
    phase[pupil]=result.x[6:]
    opd=(phase-np.mean(phase[pupil]))*wavelength/(2*np.pi)
    model=np.full(observed.shape,np.nan)
    model[:,pupil]=predicted
    phase_information=np.sum((response[None,:]*np.sin(mu-phase[pupil][None,:]-probes)/err)**2,axis=0)
    uncertainty=np.full(clear.shape,np.nan)
    uncertainty[pupil]=np.divide(1,np.sqrt(phase_information),out=np.full(n,np.inf),where=phase_information>0)
    hit=np.abs(phase[pupil])>=args.phase_bound*.99
    bound_image=np.zeros(clear.shape,np.uint8)
    bound_image[pupil]=hit
    refined=result.x[:6].reshape(2,3)
    affine_bound=bool(np.any(np.minimum(result.x[:6]-lower[:6],upper[:6]-result.x[:6])<.01*limits))
    # Convert the common phase map to a candidate correction using ALL active
    # actuators, not only the acquisition modes. Remove piston in the weighted
    # least-squares system, regularise weak actuator combinations, and constrain
    # both flat-channel and summed-DM headroom. No command is applied here.
    usable=np.isfinite(uncertainty[pupil]) & ~hit
    correction_pupil=np.zeros(clear.shape,bool)
    correction_pupil[pupil]=usable
    dm_offset=np.full((12,12),np.nan)
    dm_offset[~active]=0
    dm_offset_raw=dm_offset.copy()
    passband=np.zeros((12,12),bool)
    radial_frequency=np.full((12,12),np.nan)
    dm_opd=np.full(clear.shape,np.nan)
    corrected_opd=np.full(clear.shape,np.nan)
    dm_report=dict(success=False,reason='Fewer than 12 usable phase pixels',
                   regularization=args.dm_regularization,max_command=args.dm_max_command,
                   neighbour_regularization=DM_NEIGHBOUR_REGULARIZATION,smoothing_sigma_pitches=DM_SMOOTH_SIGMA_PITCHES,
                   applied=False,flat_recorded=flat is not None,ptt_constrained=True,
                   ptt_convention='Uniform least-squares plane over PUPIL: [1, (column-cx)/radius, (row-cy)/radius]; coefficients in nm OPD. Tip=x, tilt=y. Includes phase-bound pixels; excludes bad pixels.',
                   coldstop_diameter_lamD=COLDSTOP_DIAM_LAMD,coldstop_cutoff_cycles_per_pupil=COLDSTOP_DIAM_LAMD/2,
                   filter_convention='Direct solve in a band-limited, missing-corner and pupil-PTT null space; Gaussian spectral and neighbour penalties inside objective; fractional gain only after solve.')
    if np.count_nonzero(usable)>=12:
        uv=np.linalg.solve(refined[:,:2],(pixels-refined[:,2]).T).T
        influence=np.exp(-np.sum((uv[:,None,:]-actuators[None,:,:])**2,axis=2)/coupling**2)
        response_dm=(2*np.pi*gain/wavelength)*influence
        # Geometric, uniformly weighted pupil, not the phase-fit confidence weights.
        # Evaluate the influence functions even on pixels excluded from the HO solve.
        plane=np.column_stack([np.ones(len(pixels)),(pixels[:,0]-cx)/radius,(pixels[:,1]-cy)/radius])
        plane_inverse=np.linalg.pinv(plane)
        opd_ptt=plane_inverse@(gain*influence)
        weights=1/np.maximum(uncertainty[pupil][usable],.05)
        weights/=np.max(weights)
        H=response_dm[usable]
        target=-phase[pupil][usable]
        # Fit higher order content: marginalise piston/tip/tilt as nuisance terms.
        # The final physical zero-PTT constraint is imposed after smoothing below.
        nuisance=plane[usable]*weights[:,None]
        H=H*weights[:,None]
        target=target*weights
        H-=nuisance@np.linalg.lstsq(nuisance,H,rcond=None)[0]
        target-=nuisance@np.linalg.lstsq(nuisance,target,rcond=None)[0]
        low=np.maximum(-args.dm_max_command,-combined[active])
        high=np.minimum(args.dm_max_command,1-combined[active])
        if flat is not None:
            low=np.maximum(low,-flat[active])
            high=np.minimum(high,1-flat[active])
        if np.any(low>high):
            raise ValueError('Recorded flat/combined DM leaves incompatible correction bounds')
        free=high-low>1e-12
        if np.any(free):
            indices=np.full((12,12),-1)
            indices[active]=np.arange(active.sum())
            pairs=[]
            for left,right in [(indices[:,:-1],indices[:,1:]),(indices[:-1,:],indices[1:,:])]:
                valid=(left>=0)&(right>=0)
                pairs.extend(zip(left[valid],right[valid]))
            gradient=np.zeros((len(pairs),active.sum()))
            for row,(left,right) in enumerate(pairs):
                gradient[row,left]=1
                gradient[row,right]=-1
            dm_offset_raw,passband,radial_frequency,solution=constrained_command_fit(
                H,target,gradient,refined,2*radius,opd_ptt,low,high,args.dm_regularization)
            correction_gain=getattr(args,'gain',CORRECTION_GAIN)
            dm_offset=correction_gain*dm_offset_raw
            filter_scale=1.  # no filtering/rescaling of the optimised shape
            command=dm_offset[active]
            raw_phase=response_dm@dm_offset_raw[active]
            raw_residual=opd[pupil]+(raw_phase-np.mean(raw_phase))*wavelength/(2*np.pi)
            correction_phase=response_dm@command
            dm_opd[pupil]=correction_phase*wavelength/(2*np.pi)
            corrected_opd=opd+dm_opd
            at_limit=np.minimum(command-low,high-command)<1e-6
            before_rms=float(np.std(opd[correction_pupil])*1e9)
            after_rms=float(np.std(corrected_opd[correction_pupil])*1e9)
            dm_report.update(success=bool(solution.success),reason=solution.message,
                             correction_gain=correction_gain,command_solve='constrained_command_space',
                             final_ptt_coefficients_nm=(opd_ptt@command*1e9).tolist(),
                             raw_ptt_coefficients_nm=(opd_ptt@dm_offset_raw[active]*1e9).tolist(),
                             measured_opd_rms_nm=before_rms,predicted_residual_rms_nm=after_rms,
                             max_abs_command=float(np.max(np.abs(command))),bound_actuators=int(np.sum(at_limit)),
                             useful_pixels=int(usable.sum()),filter_headroom_scale=filter_scale,
                             neighbour_difference_rms=float(np.sqrt(np.mean((gradient@command)**2))),
                             neighbour_difference_max=float(np.max(np.abs(gradient@command))),
                             full_gain_predicted_residual_rms_nm=float(np.std(raw_residual[usable])*1e9),
                             removed_command_rms=float(np.sqrt(np.mean((dm_offset_raw[active]-command)**2))),
                             candidate_valid=bool(solution.success and result.success and not affine_bound and after_rms<before_rms),
                             baseline_note='Add DM_OFFSET to the recorded channel-0 flat with other baseline channels unchanged. A changed baseline requires a new estimate.')
        else:
            dm_report['reason']='No actuator headroom for a correction'
    report=dict(success=bool(result.success),message=result.message,nfev=result.nfev,
                calibration_reused=cached_registration is not None,
                atmospheric_repeat_groups=repeated_groups,
                uncertainty_model='Maximum of within-block/batch mean error and repeat-cycle scatter, including paired differences; diagonal approximation, not full atmospheric covariance.',
                phase_bound_rad=args.phase_bound,phase_bound_pixels=int(hit.sum()),registration_bound_hit=affine_bound,
                residual_rms_normalised=float(np.sqrt(np.mean((obs-predicted)**2))),
                weighted_residual_rms=float(np.sqrt(np.mean(((obs-predicted)/err)**2))),
                opd_rms_nm=float(np.std(opd[pupil])*1e9),background_adu=background,normalisation_adu=norm,
                pupil_geometry=[cx,cy,radius],coupling=coupling,opd_per_cmd=gain,mu_rad=mu,
                registration_prior_px=args.registration_prior_px,
                corner_prior_px=corner_prior.tolist(),bad_pixel_count=int(bad_pixels.sum()),
                modal_registration_cost=float(best.cost),modal_registration_success=bool(best.success),
                corner_disagreement_px=corner_disagreement.tolist(),
                registration_pair_gap_median_s=float(np.median(pair_gaps)) if pair_gaps else None,
                registration_pair_gap_max_s=float(np.max(pair_gaps)) if pair_gaps else None,
                dm_correction=dm_report,
                registration_initial_rms_px=float(np.sqrt(np.mean((corner_design@affine.T-centres)**2))),
                registration_refined_rms_px=float(np.sqrt(np.mean((corner_design@refined.T-centres)**2))),
                assumption='Single-wavelength local intensity law; fixed ideal circular-pupil radial reference amplitude; one shared phase map relative to reference phase zero. OPD piston removed. PHASE_SIGMA is conditional on fixed registration/model, not total uncertainty.',
                input=str(args.input.resolve()))
    header=fits.Header()
    header['WLREF']=wavelength
    header['SUCCESS']=result.success
    header['MODEL']='ANALYTIC'
    header['VERSION']=8
    header['CSDIAM']=COLDSTOP_DIAM_LAMD
    header['CSCUT']=COLDSTOP_DIAM_LAMD/2
    header['CORGAIN']=getattr(args,'gain',CORRECTION_GAIN)
    header['DMPTT']=True
    header['DMSIGMA']=DM_SMOOTH_SIGMA_PITCHES
    header['DMGRAD']=DM_NEIGHBOUR_REGULARIZATION
    hdus=[fits.PrimaryHDU(header=header),json_hdu('FIT_REPORT',report),json_hdu('SETTINGS',settings),
          json_hdu('RECORDS',[records[i] for i in selected]),
          json_hdu('FIT_OPTIONS',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})]
    for name,data in [('PHASE_RAD',phase),('OPD',opd),('PUPIL',pupil.astype(np.uint8)),('CLEAR_MEAN',clear_measured),('A',A),('B_AMPLITUDE',B),
                      ('OBSERVED',observed),('MODEL',model),('RESIDUAL',observed-model),('SIGMA',sigma),('SIGMA_WITHIN',within),('SIGMA_ATMOS',atmosphere),
                      ('PHASE_SIGMA',uncertainty),('PHASE_BOUND',bound_image),('CORNER_IMAGES',np.asarray(corner_images)),
                      ('CORNER_DIFFERENCE_SE',np.asarray(corner_errors)),('CORNER_DM',corners),('CORNER_CENTRES',centres),('CORNER_GAUSSIANS',np.asarray(gaussian_parameters)),
                      ('BAD_PIXELS',bad_pixels.astype(np.uint8)),('AFFINE_CORNERS',corner_affine),('DIAGONAL_CENTRE',centre),('AFFINE_INITIAL',affine),('AFFINE_REFINED',refined),
                      ('DM_OFFSET_RAW',dm_offset_raw),('DM_FILTER_PASSBAND',passband.astype(np.uint8)),
                      ('DM_FREQ_CPP',radial_frequency),('DM_OFFSET',dm_offset),('DM_BEFORE',combined),('CHANNEL_BEFORE',before),('DM_CORRECTION_OPD',dm_opd),('PREDICTED_RESIDUAL_OPD',corrected_opd),
                      ('DM_FIT_PUPIL',correction_pupil.astype(np.uint8))]:
        hdu=fits.ImageHDU(data,name=name)
        if name=='OPD' or name.endswith('_OPD'):
            hdu.header['BUNIT']='m'
        if name in ('DM_OFFSET','DM_OFFSET_RAW'):
            hdu.header['BUNIT']='DM command'
        if name=='DM_OFFSET':
            hdu.header['COMMENT']='Constrained fractional correction; application requires terminal approval.'
        if name=='DM_OFFSET_RAW':
            hdu.header['COMMENT']='Full-gain constrained solution before fractional gain; retained extension name for compatibility.'
        if name=='DM_FREQ_CPP':
            hdu.header['BUNIT']='cycles/pupil'
        if name=='CLEAR_MEAN':
            hdu.header['BUNIT']='ADU'
        if name=='DM_FILTER_PASSBAND':
            hdu.header['COMMENT']='Unshifted numpy FFT ordering on 12x12 DM command grid.'
        if name in ('PHASE_RAD','PHASE_SIGMA'):
            hdu.header['BUNIT']='rad'
        hdus.append(hdu)
    if flat is not None:
        for name,array in [('FLAT_BEFORE',flat),('FLAT_CANDIDATE',flat+dm_offset)]:
            hdu=fits.ImageHDU(array,name=name)
            hdu.header['BUNIT']='DM command'
            hdus.append(hdu)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fits.HDUList(hdus).writeto(args.output,checksum=True,overwrite=args.overwrite)
    print(json.dumps(report,indent=2))
    print(f'Fit saved: {args.output}')
    if not result.success or np.any(hit) or affine_bound or not dm_report.get('candidate_valid',False):
        print('FIT/CORRECTION NEEDS REVIEW: inspect convergence, parameter bounds and predicted residual diagnostics.')


def correction_warnings(report):
    """Fit-quality advice, never a veto on an explicit operator approval."""
    warnings=[]
    dm=report.get('dm_correction',{})
    if report.get('atmospheric_repeat_groups',1)==0:
        warnings.append('No repeated modal cycles: atmospheric uncertainty is unmeasured.')
    if not report.get('success',True):
        warnings.append('Phase fit did not converge: '+str(report.get('message','')))
    if report.get('phase_bound_pixels',0):
        warnings.append(f"Phase bound reached at {report['phase_bound_pixels']} pixels.")
    if report.get('registration_bound_hit',False):
        warnings.append('Registration refinement reached its allowed bounds.')
    if not report.get('modal_registration_success',True):
        warnings.append('Modal registration search reached its iteration limit; inspect the overlay.')
    if max(report.get('corner_disagreement_px',[0]))>1:
        warnings.append('Corner and modal registration disagree by more than one pixel; corner priors were weakened. Inspect the registration overlay.')
    if not dm.get('success',True):
        warnings.append('DM correction solver needs review: '+str(dm.get('reason','')))
    before=dm.get('measured_opd_rms_nm')
    after=dm.get('predicted_residual_rms_nm')
    if before is not None and after is not None and after>=before:
        warnings.append(f'Predicted residual {after:.3f} nm is not lower than measured OPD {before:.3f} nm (model prediction only).')
    if dm.get('bound_actuators',0):
        warnings.append(f"Correction reaches command bounds on {dm['bound_actuators']} actuators.")
    if dm.get('filter_headroom_scale',1)<1:
        warnings.append(f"Filtered correction was scaled by {dm['filter_headroom_scale']:.3g} to respect command limits.")
    if not dm.get('candidate_valid',True) and not warnings:
        warnings.append('Automatic fit/correction recommendation is negative; inspect the saved diagnostics.')
    return warnings


#  Save diagnostics, show the proposed offset, and apply only after 'y'.
def analyse_fit(args):
    """Static diagnostics; all per-exposure arrays remain available in the FITS."""
    import matplotlib.pyplot as plt
    with fits.open(args.input) as h:
        report=read_json(h,'FIT_REPORT')
        data={part.name:part.data.copy() for part in h if isinstance(part,fits.ImageHDU)}
    print(json.dumps(report,indent=2))
    figures={}
    valid=np.any(np.isfinite(data['RESIDUAL']),axis=0)
    residual_rms=np.full(valid.shape,np.nan)
    residual_rms[valid]=np.sqrt(np.nanmean(data['RESIDUAL'][:,valid]**2,axis=0))
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    figures['phase']=fig
    for ax,array,title in zip(axes,[data['OPD']*1e9,data['PHASE_SIGMA'],residual_rms],
                              ['Estimated OPD [nm]','Conditional phase uncertainty [rad]','Intensity residual RMS']):
        im=ax.imshow(array,origin='lower')
        ax.set_title(title)
        fig.colorbar(im,ax=ax)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    figures['dm_correction']=fig
    for ax,array,title in zip(axes,[data['DM_OFFSET'],data['DM_CORRECTION_OPD']*1e9,data['PREDICTED_RESIDUAL_OPD']*1e9],
                              ['Proposed offset [DM units]','Correction OPD [nm]','Predicted residual OPD [nm]']):
        im=ax.imshow(array,origin='lower',cmap='RdBu_r')
        ax.set_title(title)
        fig.colorbar(im,ax=ax)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    figures['registration_pupil']=fig
    clear=data['CLEAR_MEAN']
    ax.imshow(clear,origin='lower',cmap='gray',vmin=np.nanpercentile(clear,1),vmax=np.nanpercentile(clear,99))
    y,x=np.indices((12,12))
    active=np.ones((12,12),bool)
    active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
    positions=np.column_stack([x[active]-5.5,y[active]-5.5,np.ones(active.sum())])@data['AFFINE_REFINED'].T
    ax.scatter(*positions.T,s=14,c='cyan',label='Actuator centres')
    ax.scatter(*data['CORNER_CENTRES'].T,c='red',marker='x',label='Corner measurements (reused on later passes)')
    ax.plot(*data['AFFINE_REFINED'][:,2],'+',color='magenta',label='DM centre')
    if 'BAD_PIXELS' in data:
        y,x=np.nonzero(data['BAD_PIXELS'])
        ax.scatter(x,y,c='yellow',s=18,marker='x',label='Excluded pixels')
    ax.set(title='DM registration in camera pixels',xlabel='Column',ylabel='Row')
    ax.legend(fontsize=7)
    # Show the first measured +/- pair as a quick response-model diagnostic.
    with fits.open(args.input) as h:
        records=read_json(h,'RECORDS')
    positive=next((i for i,r in enumerate(records) if r['sign']==1),None)
    if positive is not None:
        r=records[positive]
        negative=next((i for i,t in enumerate(records) if t['mode']==r['mode'] and t['cycle']==r['cycle'] and t['sign']==-1),None)
        if negative is not None:
            observed=data['OBSERVED'][positive]-data['OBSERVED'][negative]
            model=data['MODEL'][positive]-data['MODEL'][negative]
            fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
            figures['probe_response']=fig
            for ax,array,title in zip(axes,[observed,model,observed-model],['Measured +/- difference','Model +/- difference','Difference residual']):
                im=ax.imshow(array,origin='lower',cmap='RdBu_r')
                ax.set_title(title)
                fig.colorbar(im,ax=ax)
    if args.save_prefix:
        args.save_prefix.parent.mkdir(parents=True,exist_ok=True)
        for name,fig in figures.items():
            fig.savefig(f'{args.save_prefix}_{name}.png',dpi=150)
    if args.no_show:
        plt.close('all')
    else:
        plt.show()


def review_correction(args, fit_path, plot_prefix):
    """Save the offset, preview it, and only then offer a flat-channel write."""
    import matplotlib.pyplot as plt
    with fits.open(fit_path) as h:
        offset=h['DM_OFFSET'].data.copy()
        settings=read_json(h,'SETTINGS')
        report=read_json(h,'FIT_REPORT')
        warnings=correction_warnings(report)
        header=fits.Header(dict(BEAM=settings['beam'],CHANNEL=0,BUNIT='DM command',APPLIED=False,APPSTATE='SAVED',
                                CORGAIN=report['dm_correction'].get('correction_gain',1),DMPTT=report['dm_correction'].get('ptt_constrained',False),QUALWARN=bool(warnings),QUALOVR=False))
        header['COMMENT']='Additive offset, not the complete flat.'
        hdus=[fits.PrimaryHDU(offset,header=header),json_hdu('SETTINGS',settings),json_hdu('FIT_REPORT',report),json_hdu('QUALITY_WARNINGS',warnings)]
        for name in ('FLAT_BEFORE','FLAT_CANDIDATE','DM_BEFORE','CHANNEL_BEFORE','DM_OFFSET_RAW'):
            if name in h:
                hdus.append(fits.ImageHDU(h[name].data.copy(),name=name))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fits.HDUList(hdus).writeto(args.output,overwrite=args.overwrite,checksum=True)
    if args.save_plots:
        analyse_fit(argparse.Namespace(input=fit_path,save_prefix=plot_prefix,no_show=True))
    display=offset.copy()
    display[0,0]=display[0,-1]=display[-1,0]=display[-1,-1]=np.nan
    limit=max(float(np.nanmax(np.abs(display))) if np.any(np.isfinite(display)) else 0,1e-12)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    im=ax.imshow(display,origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit)
    fig.colorbar(im,ax=ax,label='Additive DM command')
    ax.set_title(f'Beam {settings["beam"]}: proposed flat offset')
    if args.save_plots:
        fig.savefig(f'{plot_prefix}_preview.png',dpi=150)
    print(f'Offset saved: {args.output}. Close the preview to answer in the terminal.')
    try:
        plt.show(block=True)
        for warning in warnings:
            print('WARNING:',warning)
        answer=input("Add this offset to flat channel 0? Type 'y' to apply: ") if np.isfinite(offset).all() else ''
    except (EOFError,KeyboardInterrupt):
        answer=''
    finally:
        plt.close(fig)
    if answer.strip().lower()!='y':
        with fits.open(args.output,mode='update') as h:
            h[0].header['APPSTATE']='NOT_APPLIED'
        print('Flat unchanged.')
        return

    from xaosim.shmlib import shm
    beam=settings['beam']
    with fits.open(args.output) as h:
        expected={'flat':h['FLAT_BEFORE'].data.copy(),'master':h['DM_BEFORE'].data.copy(),'probe':h['CHANNEL_BEFORE'].data.copy()}
    paths={'flat':args.shm_dir/f'dm{beam}disp00.im.shm','master':args.shm_dir/f'dm{beam}.im.shm',
           'probe':args.shm_dir/f'dm{beam}disp{settings["channel"]:02d}.im.shm'}
    with ExitStack() as stack:
        streams={}
        for name,path in paths.items():
            stat=path.stat()
            identity=(stat.st_dev,stat.st_ino)
            if tuple(settings.get('shm_identity',{}).get(name,identity))!=identity:
                raise RuntimeError('SHM replaced since acquisition')
            stream=shm(str(path),nosem=name!='master')
            streams[name]=stream
            stack.callback(stream.close,erase_file=False)
            for sem in getattr(stream,'sems',[]):
                stack.callback(sem.close)
            opened=os.fstat(stream.fd)
            if (opened.st_dev,opened.st_ino)!=identity:
                raise RuntimeError('SHM replaced while opening')
            if not np.array_equal(stream.get_data(),expected[name]):
                raise RuntimeError(f'{name} command changed since acquisition')
        flat=streams['flat']
        master=streams['master']
        before=flat.get_data().copy()
        candidate=(before+offset).astype(before.dtype)
        active=np.ones((12,12),bool)
        active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
        if candidate.shape!=(12,12) or np.any(offset[~active]!=0):
            raise ValueError('Invalid offset shape/corners')
        combined=expected['master']+(candidate.astype(float)-before)
        if not np.isfinite(candidate).all() or np.any((candidate[active]<0)|(candidate[active]>1)|(combined[active]<0)|(combined[active]>1)):
            raise ValueError('Flat or combined DM would exceed command bounds')
        # Recheck immediately before writing; only channel 0 is changed.
        for name,path in paths.items():
            current=path.stat();opened=os.fstat(streams[name].fd)
            if (current.st_dev,current.st_ino)!=(opened.st_dev,opened.st_ino):
                raise RuntimeError('SHM replaced before application')
        if any(not np.array_equal(streams[name].get_data(),value) for name,value in expected.items()):
            raise RuntimeError('DM changed before application')
        with fits.open(args.output,mode='update') as h:
            h[0].header['APPSTATE']='WRITE_PENDING'
            h[0].header['QUALOVR']=bool(warnings)
            h.flush()
            try:
                flat.set_data(candidate)
                master.post_sems(DM_UPDATE_SEMID)
                deadline=time.monotonic()+args.timeout
                tolerance=8*np.finfo(np.dtype(master.npdtype)).eps
                while not np.allclose(master.get_data(),combined,rtol=0,atol=tolerance):
                    if time.monotonic()>deadline:
                        raise TimeoutError('Flat written but combined DM did not acknowledge it')
                    time.sleep(.002)
                if not np.array_equal(flat.get_data(),candidate):
                    raise RuntimeError('Flat readback differs')
                h.append(fits.ImageHDU(candidate,name='FLAT_APPLIED'))
                report['dm_correction']['applied']=True
                h[h.index_of('FIT_REPORT')]=json_hdu('FIT_REPORT',report)
                h[0].header['APPLIED']=True
                h[0].header['APPSTATE']='APPLIED'
                h[0].header['APPTIME']=time.time()
            except BaseException:
                report['dm_correction']['applied']=None
                h[h.index_of('FIT_REPORT')]=json_hdu('FIT_REPORT',report)
                h[0].header['APPSTATE']='UNCERTAIN'
                h.flush()
                raise
    print(f'Applied beam {beam} offset to flat channel 0.')


# 4. Repeat acquisition -> fit -> approval (one pass by default).
def run_workflow(args):
    iterations=getattr(args,'iterations',MAX_CORRECTION_ITERATIONS)
    stem=args.output.with_suffix('')
    if not args.execute:
        acquire(args)
        print(f'Up to {iterations} approved corrections at gain {getattr(args,"gain",CORRECTION_GAIN)}; no extra validation acquisitions.')
        print('Later passes reuse initial clear/corner calibration and measure fresh modal probes. Every write requires y.')
        return
    proposals=[Path(f'{stem}_iter{i+1:02d}.fits') for i in range(iterations)]
    manifest_path=Path(f'{stem}_iterations.json')
    outputs=[args.output,manifest_path]
    for proposal in proposals:
        prefix=proposal.with_suffix('')
        outputs.append(proposal)
        if args.save_intermediates:
            outputs += [Path(f'{prefix}_probes.fits'),Path(f'{prefix}_probes.fits.partial'),Path(f'{prefix}_fit.fits')]
        if args.save_plots:
            outputs += list(prefix.parent.glob(prefix.name+'_*.png'))
    if not args.overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(f'{path} exists; use --overwrite')
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest=dict(gain=getattr(args,'gain',CORRECTION_GAIN),maximum_iterations=iterations,status='running',
                  note='Fresh modal measurements per pass, reused initial calibration. Trend is not held-out validation; last update is unmeasured until another pass.',iterations=[])
    def checkpoint():
        manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    checkpoint()
    cache=None
    previous=None
    expected_flat=None
    initial_baseline=None
    applied_files=[]
    try:
        with tempfile.TemporaryDirectory(prefix='dm_probe_') as temporary:
            for number,proposal in enumerate(proposals,1):
                prefix=proposal.with_suffix('')
                item=dict(iteration=number,proposal=str(proposal),applied=False,stage='acquiring')
                manifest['iterations'].append(item)
                checkpoint()
                base=prefix if args.save_intermediates else Path(temporary)/proposal.stem
                raw_path=Path(f'{base}_probes.fits')
                fit_path=Path(f'{base}_fit.fits')
                acquisition=argparse.Namespace(**vars(args))
                acquisition.command='acquire'
                acquisition.output=raw_path
                acquisition.calibration_cache=cache
                acquisition.expected_flat=expected_flat
                acquire(acquisition)
                item['stage']='fitting'
                checkpoint()
                fitting=argparse.Namespace(**vars(args))
                fitting.command='fit'
                fitting.input=raw_path
                fitting.output=fit_path
                fit_file(fitting)
                with fits.open(fit_path) as h:
                    report=read_json(h,'FIT_REPORT')
                    opd=h['OPD'].data.copy()
                    pupil=h['PUPIL'].data.astype(bool)&~h['PHASE_BOUND'].data.astype(bool)&np.isfinite(opd)
                    y,x=np.indices(pupil.shape)
                    def ho_rms(image,mask):
                        T=np.column_stack([np.ones(mask.sum()),x[mask],y[mask]])
                        values=image[mask]*1e9
                        residual=values-T@np.linalg.lstsq(T,values,rcond=None)[0]
                        return float(np.sqrt(np.mean(residual**2)))
                    item['measured_ho_rms_nm']=ho_rms(opd,pupil) if pupil.sum()>=12 else None
                    if previous is not None:
                        common=pupil&previous[1]
                        if common.sum()>=12:
                            old=ho_rms(previous[0],common)
                            new=ho_rms(opd,common)
                            item['trend']=dict(previous_ho_rms_nm=old,current_ho_rms_nm=new,common_pixels=int(common.sum()))
                            print(f'Fresh measurement after previous update: HO RMS {old:.2f} -> {new:.2f} nm (common pixels; trend only).',flush=True)
                    previous=(opd,pupil)
                    if REUSE_ITERATION_CALIBRATION:
                        if cache is None:
                            with fits.open(raw_path) as raw:
                                settings=read_json(raw,'SETTINGS')
                                cache=dict(source=str(raw_path),clear_frames=raw['CLEAR_FRAMES'].data.copy(),clear_record=settings['clear_record'],
                                           shm_identity=settings['shm_identity'],camera_identity=settings['camera_identity'])
                            cache['registration']=dict(centres=h['CORNER_CENTRES'].data.tolist(),gaussians=h['CORNER_GAUSSIANS'].data.tolist(),
                                images=h['CORNER_IMAGES'].data.tolist(),errors=np.where(np.isfinite(h['CORNER_DIFFERENCE_SE'].data),h['CORNER_DIFFERENCE_SE'].data,None).tolist())
                        cache['registration']['affine']=h['AFFINE_REFINED'].data.tolist()
                item['stage']='review'
                checkpoint()
                review_args=argparse.Namespace(**vars(args))
                review_args.output=proposal
                review_correction(review_args,fit_path,prefix)
                with fits.open(proposal) as h:
                    item['applied']=bool(h[0].header.get('APPLIED',False))
                    if item['applied']:
                        expected_flat=h['FLAT_APPLIED'].data.copy()
                        if initial_baseline is None:
                            initial_baseline=h['FLAT_BEFORE'].data.copy()
                        applied_files.append(str(proposal))
                        # Canonical output remains usable by the no-argument reapply
                        # script after a reset: original baseline + ALL approved steps.
                        cumulative=expected_flat.astype(float)-initial_baseline.astype(float)
                        header=fits.Header()
                        header['BEAM']=args.beam
                        header['CHANNEL']=0
                        header['BUNIT']='DM command'
                        header['APPLIED']=True
                        header['APPSTATE']='APPLIED'
                        header['NSTEPS']=len(applied_files)
                        header['STEPGAIN']=getattr(args,'gain',CORRECTION_GAIN)
                        header['COMMENT']='Cumulative approved offset. Each increment, not this sum under changing registration, is constrained in pupil OPD.'
                        summary=dict(dm_correction=dict(applied=True,cumulative=True),step_files=applied_files,
                                     note='Per-step fit/constraint diagnostics are in step files; cumulative PTT is not re-projected.')
                        fits.HDUList([fits.PrimaryHDU(cumulative,header=header),json_hdu('SETTINGS',read_json(h,'SETTINGS')),
                            json_hdu('FIT_REPORT',summary),fits.ImageHDU(initial_baseline,name='FLAT_BEFORE'),
                            fits.ImageHDU(expected_flat,name='FLAT_CANDIDATE'),fits.ImageHDU(expected_flat,name='FLAT_APPLIED')
                            ]).writeto(args.output,overwrite=True,checksum=True)
                    elif initial_baseline is None:
                        fits.HDUList([part.copy() for part in h]).writeto(args.output,overwrite=True,checksum=True)
                item['stage']='applied' if item['applied'] else 'declined'
                checkpoint()
                if not item['applied']:
                    manifest['status']='declined'
                    break
            else:
                manifest['status']='completed'
    except BaseException as exc:
        manifest['status']='interrupted_or_failed'
        manifest['error']=str(exc)
        if manifest['iterations']:
            item=manifest['iterations'][-1]
            if item['stage']=='review' and Path(item['proposal']).exists():
                with fits.open(item['proposal']) as h:
                    item['apply_state']=h[0].header.get('APPSTATE','UNKNOWN')
                    item['applied']=None if item['apply_state'] in ('UNCERTAIN','WRITE_PENDING') else bool(h[0].header.get('APPLIED',False))
        print('Sequence stopped; inspect the iteration record. No automatic rollback was attempted.')
        raise
    finally:
        checkpoint()
    print(f'Iteration record saved: {manifest_path}; accumulated approved offset: {args.output}')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    commands=parser.add_subparsers(dest='command',required=True)
    defaults={**ACQUISITION_DEFAULTS,**FIT_DEFAULTS}
    for name in ('run','acquire','fit','analyse'):
        sub=commands.add_parser(name)
        sub.set_defaults(**defaults)
        if name in ('run','acquire'):
            sub.add_argument('--beam',type=int,choices=range(1,5),required=True)
            sub.add_argument('--amplitude',type=float,required=True,help='Modal probe amplitude in DM units')
            sub.add_argument('--registration-amplitude',type=float,default=.1)
            sub.add_argument('--basis',choices=['Mixed','Zernike','Fourier','FourierModified','Zonal'])
            sub.add_argument('--modes',type=int,nargs='+')
            sub.add_argument('--frames',type=int,default=10)
            sub.add_argument('--cycles',type=int,default=2)
            sub.add_argument('--mask',choices=MASKS,default='H3')
            sub.add_argument('--simulator',action='store_true',default=SIMULATION)
            sub.add_argument('--execute',action='store_true',help='Without this, print the acquisition plan only')
            
        if name in ('run','fit'):
            sub.add_argument('--gain',type=float,default=CORRECTION_GAIN)
            
        if name=='run':
            sub.add_argument('--iterations',type=int,default=MAX_CORRECTION_ITERATIONS)
            sub.add_argument('--save-plots',action=argparse.BooleanOptionalAction,default=True)
            sub.add_argument('--save-intermediates',action=argparse.BooleanOptionalAction,default=True)
            sub.add_argument('--save-prefix',type=Path)
        if name in ('fit','analyse'):
            sub.add_argument('input',type=Path)
        if name=='fit':
            sub.set_defaults(coupling=None,opd_per_cmd=None)  # use recorded calibration
        if name=='analyse':
            sub.add_argument('--save-prefix',type=Path)
            sub.add_argument('--no-show',action='store_true')
        else:
            sub.add_argument('--output',type=Path,required=True)
            sub.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    if args.command in ('run','acquire'):
        args.mds=f'tcp://{MDS_HOST_SIMULATOR if args.simulator else MDS_HOST_ONSKY}:5555'
        if args.basis is None:
            args.basis='Mixed' if args.modes is None else 'Zernike'
        if args.modes is None:
            args.modes=list(range(len(MIXED_LABELS))) if args.basis=='Mixed' else [0,1,2,3]
        if not np.isfinite([args.amplitude,args.registration_amplitude]).all() or min(args.amplitude,args.registration_amplitude)<=0:
            parser.error('Probe amplitudes must be finite and positive')
        if args.frames<2 or args.cycles<1:
            parser.error('frames must be >=2; cycles must be >=1')
        if min(args.modes)<0 or len(set(args.modes))!=len(args.modes):
            parser.error('Mode indices must be unique and nonnegative')
        if args.basis=='Mixed' and max(args.modes)>=len(MIXED_LABELS):
            parser.error('Mixed mode indices are 0–10')
    if args.command in ('run','fit') and (not np.isfinite(args.gain) or not 0<args.gain<=1):
        parser.error('gain must be in (0,1]')
    if args.command=='run' and args.iterations<1:
        parser.error('iterations must be >=1')
    {'run':run_workflow,'acquire':acquire,'fit':fit_file,'analyse':analyse_fit}[args.command](args)


if __name__=='__main__':
    main()
