import numpy as np
from scipy.interpolate import griddata
from xaosim.zernike import mkzer1
from xaosim.pupil import _dist as dist


import minimal_baldr_python_rtc.DM as DM
import minimal_baldr_python_rtc.utils as utils
import minimal_baldr_python_rtc.Cam as Cam

ndm = 4  # the number of DMs to control
nzer = 11  # the number of Zernike modes to control
dms = 12  # the DM size
aps = 10  # the aperture grid size


dd = dist(dms, dms, between_pix=True)  # auxilliary array
tprad = 5.5  # the taper function radius
taper = np.exp(-((dd / tprad) ** 20))  # power to be adjusted ?
amask = taper > 0.4  # seems to work well
circ = dd < 4


# functions copied from MDM controller
def fill_mode(dmmap):
    """Extrapolate the modes outside the aperture to ensure edge continuity

    Parameter:
    ---------
    - a single 2D DM map
    """
    out = True ^ amask  # outside the aperture
    gx, gy = np.mgrid[0:dms, 0:dms]
    points = np.array([gx[amask], gy[amask]]).T
    values = np.array(dmmap[amask])
    grid_z0 = griddata(points, values, (gx[out], gy[out]), method="nearest")
    res = dmmap.copy()
    res[out] = grid_z0
    return res


def zer_bank(i0, i1, extrapolate=True):
    """------------------------------------------
    Returns a 3D array containing 2D (dms x dms)
    maps of Zernike modes for Noll index going
    from i0 to i1 included.

    Parameters:
    ----------
    - i0: the first Zernike index to be used
    - i1: the last Zernike index to be used
    - tapered: boolean (tapers the Zernike)
    ------------------------------------------"""
    dZ = i1 - i0 + 1
    res = np.zeros((dZ, dms, dms))
    for ii in range(i0, i1 + 1):
        test = mkzer1(ii, dms, aps // 2, limit=False)
        # if ii == 1:
        #     test *= circ
        if ii != 1:
            test -= test[amask].mean()
            test /= test[amask].std()
        if extrapolate is True:
            # if ii != 1:
            test = fill_mode(test)
        res[ii - i0] = test

    return res


beams = [1, 2, 3, 4]
focus = zer_bank(4, 4)
probeDMs = [
    DM.DM(i, main_chn=1, basis=focus) for i in beams
]  # the dm we will apply focus probes on
DMs = [DM.FourierDM(i) for i in beams]
probe_ch = 1
# cams = [Cam.Cam(i) for i in beams]


def run_fdpr_single(
    beam,
    cam,
    probe_dm,
    aber_dm,
    max_foc_amp=0.7,
    n_foc_steps=50,
    sleep=0.01,
    n_discard=1,
    n_im=100,
):
    ctx, sock = utils.mds_connect("mimir", 5555)

    try:
        probe_dm.flatten()
        aber_dm.flatten()

        bmy_pos = utils.mds_send(sock, f"read BMY{beam}")
        utils.mds_send(sock, f"moveabs BMY{beam} 500.0")
        time.sleep(3)

        cam.take_dark(256)

        cam_dark = cam.dark.copy()
        dark = cam.take_stack(1000)

        utils.mds_send(sock, f"moveabs BMY{beam} {bmy_pos}")
        time.sleep(3)

        # take img only reference
        ref = cam.take_stack(2000)

    finally:
        pass
