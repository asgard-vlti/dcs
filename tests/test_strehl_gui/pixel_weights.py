"""Conservative clear-reference pixel quality weights; no live adaptive clipping."""
import numpy as np
from scipy.ndimage import generic_filter


def clear_pixel_weights(cube,pupil,read_noise_e,background_e=0.):
    cube=np.asarray(cube,float)
    pupil=np.asarray(pupil,bool)
    if cube.ndim!=3 or cube.shape[1:]!=pupil.shape or len(cube)<3:
        raise ValueError('Pixel weighting needs >=3 clear frames and matching pupil')
    finite=np.all(np.isfinite(cube),axis=0)
    safe=np.where(np.isfinite(cube),cube,0.)
    mean=safe.mean(axis=0)
    variance=safe.var(axis=0,ddof=1)
    expected=np.maximum(mean,0)+read_noise_e**2+background_e
    weights=np.zeros(pupil.shape)
    mean_score=np.zeros(pupil.shape)
    variance_score=np.zeros(pupil.shape)
    def local_median(values):
        values=values[np.isfinite(values)]
        return np.median(values) if len(values) else np.nan
    for region in (pupil,~pupil):
        good=region & finite
        if good.sum()<4:
            raise ValueError('Too few finite clear pixels inside or outside pupil')
        local=generic_filter(np.where(good,mean,np.nan),local_median,size=5,mode='constant',cval=np.nan)
        residual=mean-local
        centre=np.median(residual[good])
        spread=1.4826*np.median(np.abs(residual[good]-centre))
        # Include noise of the finite-frame mean; floor avoids zero MAD instability.
        scale=np.maximum(spread,np.sqrt(np.maximum(expected,1.)/len(cube)))
        z=np.abs(residual-centre)/np.maximum(scale,1e-6)
        # Compare variance after allowing for different shot noise across illumination.
        logratio=np.log(np.maximum(variance,1e-6)/np.maximum(expected,1e-6))
        median=np.median(logratio[good])
        mad=1.4826*np.median(np.abs(logratio[good]-median))
        v=np.abs(logratio-median)/max(mad,np.sqrt(2/(len(cube)-1)),.15)
        mean_score[good]=z[good]
        variance_score[good]=v[good]
        score=np.maximum(z/8.,v/6.)
        # Keep normal pixels exactly at 1; smoothly downweight suspicious pixels.
        w=np.minimum(1.,1/np.maximum(score,1.)**4)
        w[score>=3]=0.
        weights[good]=w[good]
    return dict(weights=weights,mean_e=np.where(finite,mean,np.nan),
                variance_e2=np.where(finite,variance,np.nan),
                mean_score=mean_score,variance_score=variance_score)
