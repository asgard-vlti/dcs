"""Fresh calibration captures and bounded mask-offset transactions.

No motion happens on import. Each RPC owns a fresh REQ socket so a timeout does
not leave subsequent recovery reads in an invalid REQ state.
"""
import time
import numpy as np
from .monitor import Camera


def rpc(address,command,timeout=3000):
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
        if reply.upper().startswith(('ERR','NACK','NOK','FAIL')):
            raise RuntimeError(f'{command}: {reply}')
        return reply
    finally:
        sock.close()
        ctx.term()


def position(call,axis):
    value=float(call(f'read {axis}'))
    if not np.isfinite(value):
        raise ValueError('Nonfinite mask position')
    return value


def wait_position(call,axis,target,tolerance,timeout):
    end=time.monotonic()+timeout
    while True:
        if abs(position(call,axis)-target)<=tolerance:
            return
        if time.monotonic()>=end:
            raise TimeoutError(f'Mask did not reach {target:g} on {axis}')
        time.sleep(.05)


def fresh_frames(camera,frames,timeout):
    # Flush the most recent exposure: even with a new reader it may predate settling.
    camera.read()
    rows=[]
    deadline=time.monotonic()+timeout
    while len(rows)<frames:
        sample=camera.read()
        if sample is not None:
            counter,frame=sample
            frame=np.asarray(frame,float)
            if frame.ndim!=2 or not np.any(np.isfinite(frame)):
                raise ValueError('Invalid calibration frame')
            if rows and frame.shape!=rows[0][2].shape:
                raise ValueError('Camera shape changed during calibration')
            rows.append((time.time(),counter,frame.copy()))
        elif time.monotonic()>=deadline:
            raise TimeoutError(f'Only {len(rows)}/{frames} fresh calibration frames received')
        else:
            time.sleep(.005)
    return rows


def acquire(path,frames,settle,timeout,*,mds=None,axis=None,offset=200.,
            tolerance=.1,motion_timeout=15.,camera_factory=Camera,call=None,
            progress=lambda message:None):
    """Collect fresh frames, optionally moving out and restoring a mask first.

    On ambiguous move failures, read actual position and return by the measured
    difference. Never blindly repeat a relative command. A failed restoration
    invalidates the whole acquisition and reports the original coordinate.
    """
    camera=camera_factory(path)
    original=None
    motion_attempted=False
    call=call or (lambda command:rpc(mds,command))
    try:
        try:
            if mds:
                original=position(call,axis)
                motion_attempted=True
                progress(f'Moving {axis} out by {offset:g} µm (start {original:g})')
                call(f'moverel {axis} {offset:.9g}')
                wait_position(call,axis,original+offset,tolerance,motion_timeout)
            progress(f'Settling {settle:g} s; old frames excluded')
            time.sleep(settle)
            progress(f'Collecting {frames} fresh frames')
            rows=fresh_frames(camera,frames,timeout)
        finally:
            if motion_attempted:
                progress(f'Restoring {axis} to {original:g}')
                try:
                    current=position(call,axis)
                    difference=original-current
                    if abs(difference)>tolerance:
                        call(f'moverel {axis} {difference:.9g}')
                    wait_position(call,axis,original,tolerance,motion_timeout)
                    time.sleep(settle)
                    progress('Mask restored and position verified')
                except Exception as exc:
                    raise RuntimeError(f'MASK RESTORATION UNVERIFIED: {axis} original={original:g}. '
                                       f'Check stage position before proceeding. {exc}') from exc
        return dict(rows=rows,motion=dict(axis=axis,original=original,offset=offset) if mds else None)
    finally:
        camera.close()
