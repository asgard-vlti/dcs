#!/usr/bin/env python3
"""Run without arguments to reapply a saved probe-script DM offset.

Requires numpy, astropy, xaosim and posix_ipc; no simulator/config imports.
Reads the final proposal FITS, NOT the intermediate fit or probe file.
Writes channel 0 immediately: saved FLAT_BEFORE + primary-HDU offset.
Already applied is a no-op; a different baseline is rejected to avoid stacking.
Keep other DM writers held/stopped during application. No RTC state is changed.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import time

import numpy as np
from astropy.io import fits

OFFSET_FITS = Path('dm_offset_b1.fits')  # relative to current working directory
SHM_DIR = Path('/dev/shm')


def main():
    from xaosim.shmlib import shm
    import posix_ipc

    with fits.open(OFFSET_FITS) as h:
        beam=int(h[0].header['BEAM'])
        offset=np.asarray(h[0].data,dtype=float)
        baseline=np.asarray(h['FLAT_BEFORE'].data,dtype=float)
    if beam not in (1,2,3,4) or offset.shape!=(12,12) or baseline.shape!=(12,12):
        raise ValueError('Expected final offset FITS with beam 1–4 and 12x12 FLAT_BEFORE')
    if not np.isfinite(offset).all() or not np.isfinite(baseline).all():
        raise ValueError('Non-finite saved baseline/offset')
    active=np.ones((12,12),bool)
    active[0,0]=active[0,-1]=active[-1,0]=active[-1,-1]=False
    if np.any(offset[~active]!=0):raise ValueError('Offset modifies missing DM corners')

    with ExitStack() as stack:
        paths=[SHM_DIR/f'dm{beam}disp00.im.shm',SHM_DIR/f'dm{beam}.im.shm']
        streams=[]
        identities=[]
        for path in paths:
            stat=path.stat()  # require an existing stream; never create one
            identity=(stat.st_dev,stat.st_ino)
            stream=shm(str(path),nosem=True);stack.callback(stream.close,erase_file=False)
            opened=os.fstat(stream.fd)
            if (opened.st_dev,opened.st_ino)!=identity:raise RuntimeError('SHM replaced while opening')
            streams.append(stream);identities.append(identity)
        flat,master=streams
        current=flat.get_data(check=False).copy()
        combined=master.get_data(check=False).copy()
        if current.shape!=(12,12) or combined.shape!=(12,12) or current.dtype.kind!='f':
            raise ValueError('Expected floating-point 12x12 DM streams')
        candidate=(baseline+offset).astype(current.dtype)
        candidate[~active]=current[~active]
        if np.array_equal(current,candidate):
            print(f'Beam {beam}: saved correction already applied; no change.');return
        #if not np.array_equal(current[active],baseline.astype(current.dtype)[active]):
        #    raise RuntimeError('Channel 0 differs from the saved baseline; refusing to accumulate/overwrite another flat')
        expected=combined.astype(float)+(candidate.astype(float)-current)
        if not np.isfinite(candidate).all() or not np.isfinite(expected).all():
            raise ValueError('Non-finite live/candidate DM command')
        if np.any((candidate[active]<0)|(candidate[active]>1)|(expected[active]<0)|(expected[active]>1)):
            raise ValueError('Flat or combined DM would exceed [0,1]; no change')
        prefix=str(SHM_DIR.resolve()).replace('/','.')
        sem=posix_ipc.Semaphore(f'/{prefix}.dm{beam}_sem01',flags=0);stack.callback(sem.close)
        for path,identity in zip(paths,identities):
            stat=path.stat()
            if (stat.st_dev,stat.st_ino)!=identity:raise RuntimeError('SHM replaced before write')
        if not np.array_equal(flat.get_data(check=False),current) or not np.array_equal(master.get_data(check=False),combined):
            raise RuntimeError('DM changed during preparation; hold other writers and retry')
        # Write intent first; a failed notification/readback must not claim success.
        with fits.open(OFFSET_FITS,mode='update',checksum=True) as h:
            h[0].header['RESTATE']='PENDING';h[0].header['APPSTATE']='WRITE_PENDING';h.flush()
            try:
                flat.set_data(candidate);sem.release()
                if not np.array_equal(flat.get_data(check=False),candidate):
                    raise RuntimeError('Flat readback differs after write')
                h[0].header['RESTATE']='APPLIED';h[0].header['RETIME']=time.time()
                h[0].header['APPLIED']=True;h[0].header['APPSTATE']='APPLIED'
                h[0].header['APPTIME']=h[0].header['RETIME']
                applied=fits.ImageHDU(candidate,name='FLAT_APPLIED')
                applied.header['BUNIT']='DM command';applied.add_checksum()
                if 'FLAT_APPLIED' in h:h[h.index_of('FLAT_APPLIED')]=applied
                else:h.append(applied)
                if 'FIT_REPORT' in h:
                    report=json.loads(h['FIT_REPORT'].data['JSON'][0])
                    report['dm_correction']['applied']=True
                    raw=json.dumps(report,allow_nan=False)
                    h[h.index_of('FIT_REPORT')]=fits.BinTableHDU.from_columns(
                        [fits.Column(name='JSON',format=f'{len(raw)}A',array=[raw])],name='FIT_REPORT')
            except BaseException:
                h[0].header['RESTATE']='UNCERTAIN';h[0].header['APPSTATE']='UNCERTAIN';h.flush();raise
    print(f'Beam {beam}: reapplied {OFFSET_FITS} to channel 0 (saved baseline + offset).')


if __name__=='__main__':
    main()
