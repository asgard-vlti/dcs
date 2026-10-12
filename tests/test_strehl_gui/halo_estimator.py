"""Fit-free flux diagnostic. Absolute Strehl requires an empirical calibration.

For Eout=E+(exp(i theta)-1)b, outside the incident pupil E=0 and
Iout=4 sin(theta/2)^2 |b|^2. If b=sqrt(S)b0 and the complete flux is
collected, f=Fout/(Fin+Fout) scales with S, whereas R=Fout/Fin does not.
A finite focal mask does not in general satisfy this separable-field assumption.
"""
import numpy as np


def measure_halo(cube_e, clear_e, pupil, read_noise_e, clear_frames,
                 background_e=0., calibration=None, weights=None):
    cube=np.asarray(cube_e,float)
    if cube.ndim==2:
        cube=cube[None]
    clear=np.asarray(clear_e,float)
    pupil=np.asarray(pupil,bool)
    if cube.ndim!=3 or cube.shape[1:]!=clear.shape or pupil.shape!=clear.shape:
        raise ValueError('Halo image and pupil shapes differ')
    if not pupil.any() or pupil.all():
        raise ValueError('Halo diagnostic needs inside and outside pixels')
    weights=np.ones_like(clear) if weights is None else np.asarray(weights,float)
    if weights.shape!=clear.shape or not np.all(np.isfinite(weights)) or np.any((weights<0)|(weights>1)):
        raise ValueError('Pixel weights must be finite, matching shape, in [0,1]')
    coefficients=np.zeros_like(clear)
    for region in (pupil,~pupil):
        if np.sum(weights[region])<max(1.,.5*region.sum()):
            raise ValueError('Too little reliable pixel coverage in a pupil region (<50%)')
        # Preserve area for a uniform field; missing structure remains a systematic.
        coefficients[region]=weights[region]*region.sum()/np.sum(weights[region])
    used=coefficients>0
    valid=np.all(np.isfinite(cube[:,used]),axis=1)
    cube=np.where(used,cube,0.)
    clear=np.where(used,clear,0.)
    inside=np.sum(cube[:,pupil]*coefficients[pupil],axis=1)
    outside=np.sum(cube[:,~pupil]*coefficients[~pupil],axis=1)
    total=inside+outside
    noise=(np.maximum(cube,0)+read_noise_e**2+background_e)*coefficients**2
    vi=noise[:,pupil].sum(axis=1)
    vo=noise[:,~pupil].sum(axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        ratio=outside/inside
        fraction=outside/total
        sigma=np.sqrt(inside**2*vo+outside**2*vi)/total**2
        ratio_sigma=np.sqrt(vo/inside**2+outside**2*vi/inside**4)
    valid &= (total>5*np.sqrt(vi+vo)) & (inside>5*np.sqrt(vi))
    ci,co=float((clear*coefficients)[pupil].sum()),float((clear*coefficients)[~pupil].sum())
    ct=ci+co
    if not np.isfinite(ct) or ct<=0:
        raise ValueError('Clear reference has no positive finite total flux')
    baseline=co/ct
    cn=(np.maximum(clear,0)+read_noise_e**2+background_e)*coefficients**2/max(clear_frames,1)
    baseline_sigma=float(np.sqrt(ci**2*cn[~pupil].sum()+co**2*cn[pupil].sum())/ct**2)
    excess=fraction-baseline
    excess_sigma=np.sqrt(sigma**2+baseline_sigma**2) if clear_frames>0 else np.full_like(sigma,np.nan)
    strehl=np.full_like(fraction,np.nan)
    strehl_sigma=strehl.copy()
    if calibration is not None:
        delta=calibration['fraction']-baseline
        if delta<=0:
            raise ValueError('Halo reference must exceed the clear outside fraction')
        sr=calibration['strehl']
        strehl=sr*excess/delta
        # Same clear baseline appears in numerator and denominator: keep covariance.
        strehl_sigma=abs(sr)*np.sqrt((sigma/delta)**2+
            (excess*calibration['fraction_sigma']/delta**2)**2+
            ((fraction-calibration['fraction'])*baseline_sigma/delta**2)**2)
        if clear_frames<=0:
            strehl_sigma[:]=np.nan
    for values in (ratio,fraction,sigma,ratio_sigma,excess,excess_sigma,strehl,strehl_sigma):
        values[~valid]=np.nan
    return dict(inside_e=inside,outside_e=outside,ratio=ratio,ratio_sigma=ratio_sigma,
        fraction=fraction,fraction_sigma=sigma,excess_fraction=excess,
        excess_sigma=excess_sigma,strehl=strehl,strehl_sigma=strehl_sigma,
        valid=valid,clear_fraction=baseline,clear_fraction_sigma=baseline_sigma,
        calibration=calibration,pixel_weights=weights,inside_pixels=int(pupil.sum()),outside_pixels=int((~pupil).sum()))


def reference_diagnostics(measurement):
    """Explain the calibration denominator and its conditional statistical error."""
    valid=measurement['valid']
    values=measurement['fraction'][valid]
    sigmas=measurement['fraction_sigma'][valid]
    if len(values)<3:
        raise ValueError(f'Only {len(values)}/{len(valid)} valid frames; need at least three. '
                         'Check total/pupil signal, bias, gain and read noise.')
    mean=float(values.mean())
    error=float(max(np.sqrt(np.sum(sigmas**2))/len(values),values.std(ddof=1)/np.sqrt(len(values))))
    baseline=float(measurement['clear_fraction'])
    baseline_error=float(measurement['clear_fraction_sigma'])
    delta=mean-baseline
    uncertainty=float(np.hypot(error,baseline_error))
    if not np.isfinite(uncertainty) or not np.isfinite(delta):
        raise ValueError('Nonfinite reference uncertainty; check clear reference and noise settings')
    significance=delta/uncertainty if uncertainty>0 else (float('inf') if delta>0 else 0.)
    return dict(fraction=mean,fraction_sigma=error,clear_fraction=baseline,
                clear_fraction_sigma=baseline_error,excess=delta,
                excess_sigma=uncertainty,significance=significance,frames=len(values))


def reference_from_window(measurement,strehl):
    """Reference uncertainty includes observed frame scatter; assumes independent samples."""
    if not np.isfinite(strehl) or not 0<strehl<=1:
        raise ValueError('Reference Strehl must be in (0,1]')
    d=reference_diagnostics(measurement)
    if d['excess']<=0 or d['significance']<=5:
        detail=(f"outside fraction IN={d['fraction']:.5f}, clear={d['clear_fraction']:.5f}; "
                f"excess={d['excess']:.5f} ± {d['excess_sigma']:.5f} "
                f"({d['significance']:.2f} sigma; need >5). "
                f"Uncertainty: IN={d['fraction_sigma']:.5f}, clear={d['clear_fraction_sigma']:.5f}. ")
        if d['excess']<=0:
            advice=('No positive mask-induced halo excess. Check mask OUT/IN states, '
                    'pupil circle, crop and matching source/exposure.')
        else:
            advice=('Positive excess is too uncertain. Verify detector noise/bias and matching '
                    'references; use a longer --frames window and recapture BOTH references.')
        raise ValueError(detail+advice)
    return dict(fraction=d['fraction'],fraction_sigma=d['fraction_sigma'],
                strehl=float(strehl),frames=d['frames'],significance=d['significance'])
