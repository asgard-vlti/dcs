#!/usr/bin/env python3

"""============================================================================

This is now the central piece of the data Heimdallr analysis attempts. I have
turned some of the analysis scripts worked on so far into a reusable class data
structure.

============================================================================"""

import numpy as np
import matplotlib.pyplot as plt
from xara import IWFS
from xaosim.pupil import _dist as dist
from xaosim.pupil import hex_grid_coords as hexcoords
import matplotlib.cm as cm
import astropy.io.fits as pf

# =============================================================================
#                              Cosmetic choices
# =============================================================================
plt.rcParams['image.origin'] = "lower"

# tropical summer vibes (dgrn, trk, grn, ylw, orng, dorng)
palette6 = [(38, 70, 83), (42,157,143), (138,177,125),
            (233,196,106), (244,162,97), (231,111,81)]

palette6_2 = []
for color in palette6:
    palette6_2.append((color[0]/255, color[1]/255, color[2]/255))

palette6 = palette6_2
# =============================================================================
#                                 Tools
# =============================================================================

def center_of_mass_2D(xc, yc, ww):
    """ ----------------------------------------------------------------
    Locates the center of mass of the 2D cloud of sample points,
    weighted by the values contained in ww.

    Parameters:
    - xc: coordinates along the x-axis
    - yc: coordinates along the y-axis
    - ww: the associated weight

    Returns the (x,y) location of the center of mass as a tuple.
    ---------------------------------------------------------------- """
    norm = ww.sum()
    xc0 = (xc * ww).sum() / norm
    yc0 = (yc * ww).sum() / norm
    return (xc0, yc0)

def unwrap(val, prev):
    """ ----------------------------------------------------------------
    Unwraps the phase value to ensure continuity with previous value.

    This is mostly useful for RT processing of data.

    For post-processing, use np.unwrap() an array of values
    ---------------------------------------------------------------- """
    if np.abs(val - prev) < np.pi:
        return val
    elif val > prev:
        return val - 2*np.pi
    else:
        return val + 2*np.pi

# =============================================================================

class HMDA:
    """------------------------------------------------------------------------
    HMDA: the Heimdallr Data Analysis class

    Designed to serve as the basis for the python data analysis of Heimdallr
    that is based on the XARA framework.

    The class manipulates two XARA models in parallel:
    - a sparse model - with one sample point per sub-aperture
    - a full model - with multiplt sample points per sub-aperture

    ?? Simultaneously for the two band-passes ??
    Actually not sure yet that is the best way to go. I might want to focus
    on the analysis of a single band here and see if I hit some limitations.

    The dense model is used to perform higher order analysis of the data:
    - estimate the global and differential tip-tilt of the Heimdallr beams
    - estimate the potential pupil drifts

    The sparse model is the naive model that ought to give direct visibilties
    and phase measurements.
    ------------------------------------------------------------------------"""

    def __init__(self, hc_sparse=None):
        """--------------------------------------------------------------------
        Instantiation of the class - to be completed by further functions

        Parameters:
        ----------
        - hc_sparse: 2D array of (x,y) coordinates for the holes
        --------------------------------------------------------------------"""
        self.K1_wl = 2.05e-6  # True Heimdallr Ks wavelength (in meters)
        self.K2_wl = 2.25e-6  # True Heimdallr Kl wavelength (in meters)
        self.pscale = 61.9    # should be 64.75 - adjust data simulation!
        self.isz = 32         # image size assumed by default

        # K1/2_DM2rad is to convert DM piston command amplitude into phase
        # DM documentation suggests: 3700 nm / piston ADU ("driver units")
        # experimentally, it looks closer to 3500 nm / piston ADU
        self.dm_gain = 3.5    # gain of 3500 nm / piston ADU works well
        self.K1_DM2rad = 4 * np.pi * self.dm_gain / (self.K1_wl * 1e6)
        self.K2_DM2rad = 4 * np.pi * self.dm_gain / (self.K2_wl * 1e6)
        self.COHL_mu = 2.15**2 / (2.25 - 2.05)  # coherence length (in microns)
        self.GD_ph2mu = self.COHL_mu / (2 * np.pi)  # phase to microns

        # coordinates of the model sample points
        if hc_sparse is not None:
            self.hc_sparse = hc_sparse
        else:
            self.hc_sparse = np.loadtxt("N1_hole_coordinates.txt")
        self.update_pupil_model(self.hc_sparse)

        # tools for data clean-up
        self.apm = None  # apodizing mask
        self.bgm = None  # background mask

    def update_pupil_model(self, hc_sparse):
        """--------------------------------------------------------------------
        Update the pupil model based on the provided coordinates

        Parameters:
        ----------
        - hc_sparse: 2D array of (x,y) coordinates for the holes
        --------------------------------------------------------------------"""
        print("==========>>>>> Updating pupil model!")
        self.hc_sparse = hc_sparse
        self.hc_full = []
        self.hrad = 0.25                     # Narcissus hole radius (mm)
        self.frad = 2 * self.hrad            # Fourier splodge radius (mm)
        self.nbm = self.hc_sparse.shape[0]   # number of beams (ie. 4)
        tmp = hexcoords(2, radius=0.1)       # dense sub-aperture model
        self.hexc = hexcoords(2, radius=0.1)
        self.nbsap = tmp.shape[1]            # = 19 for this use case

        for jj in range(self.nbm):
            tmp = hexcoords(2, radius=0.1)
            for ii in range(tmp.shape[1]):
                tmp[:, ii] += self.hc_sparse[jj, :]
            self.hc_full.append(tmp.T)

        self.nbt_sap = self.nbm * self.nbsap  # 4 x 19 = 76
        self.xy0 = tmp  # 
        self.hc_full = np.reshape(self.hc_full, (self.nbt_sap, 2))

        self.K1SM = IWFS(array=self.hc_sparse)  # K1 sparse model
        self.K2SM = IWFS(array=self.hc_sparse)  # K2 sparse model
        self.K1DM = IWFS(array=self.hc_full)    # K1 dense model
        # self.K2FM = IWFS(array=hc_full)

        self.sparse = self.K1SM           # alias for later convenience
        self.dense = self.K1DM            # alias for later convenience

        self.nbuv_s = self.K1SM.kpi.nbuv  # number of Fourier "splodges"
        self.nbuv_d = self.K1DM.kpi.nbuv  # number of Fourier samples points

        self.nbl = self.nbuv_s            # alias for number of baselines

        # baseline mapping matrix BLM and its inverse iBLM
        self.BLM = self.sparse.kpi.BLM
        self.iBLM = np.round(np.linalg.pinv(self.BLM), 2)

        # kernel-phase and closure-phase matrices
        self.KPM = self.sparse.kpi.KPM
        self.CPM = np.array([[ 0,  0,  0,  1, -1, -1], # hard-coded for now
                             [ 1, -1,  0,  0,  0, -1],
                             [ 1,  0, -1, -1,  0,  0],
                             [ 1,  0, -1,  1,  0,  0]])

        self.nkp = self.KPM.shape[0]  # contains 3 KPs
        self.ncp = self.CPM.shape[0]  # contains 1 additional CP

        self.K1SM.update_img_properties(
            isz=self.isz, wl=self.K1_wl, pscale=self.pscale)
        self.K2SM.update_img_properties(
            isz=self.isz, wl=self.K2_wl, pscale=self.pscale)
        self.K1DM.update_img_properties(
            isz=self.isz, wl=self.K1_wl, pscale=self.pscale)

        self.spl_ii = self.sort_full_model(self.K1DM, self.K1SM, self.frad)
        self.compute_Fourier_slope_aux_data()
        self.bl_lbls = self.make_bl_labels()
        self.cp_lbls = self.make_cp_labels()

    def make_bl_labels(self):
        """--------------------------------------------------------------------
        Make labels for the baselines that relate to the Heimdallr beam id

        Returns a list of labels in the appropriate order
        --------------------------------------------------------------------"""
        lbls = []
        for ii in range(self.nbl):
            bm1_id = np.argmax(self.BLM[ii]) + 1
            bm2_id = np.argmin(self.BLM[ii]) + 1
            lbls.append(f"(B{bm1_id}-B{bm2_id})")
        return lbls

    def make_cp_labels(self):
        """--------------------------------------------------------------------
        Make labels for the closure phases that relate to the Heimdallr beam id

        Returns a list of labels in the appropriate order
        --------------------------------------------------------------------"""
        lbls = []
        for ii in range(self.ncp):
            lbls.append(''.join(np.strings.multiply(
                self.bl_lbls, np.abs(self.CPM[ii]))))
            # bl1, bl2, bl3 = np.where(
            #     (self.CPM[ii] == -1) + (self.CPM[ii] == 1))[0]
            # lbls.append(self.bl_lbls[bl1] + " + " + \
            #             self.bl_lbls[bl2] + " + " + self.bl_lbls[bl3])
        return lbls

    def make_apodizing_mask(self, arad=6, pwr=4):
        """--------------------------------------------------------------------
        Computes a "super-Gaussian" apodizing mask of the right size
        Also makes the background mask
        --------------------------------------------------------------------"""
        dd = dist(self.isz, self.isz, between_pix=True)
        self.apm = np.exp(-(dd/arad)**pwr)
        self.bgm = self.apm < 1e-3
        return

    def apodize_data(self, data):
        """--------------------------------------------------------------------
        Procedure: apodizes the provided data.

        !! The data is modified in place !!

        Parameters:
        ----------
        - data: 2D image or 3D data cube
        --------------------------------------------------------------------"""
        if self.apm is None:
            self.make_apodizing_mask()  # default apodizer

        if data.ndim == 2:
            data *= self.apod
        elif data.ndim == 3:
            for ii in range(data.shape[0]):
                data[ii] *= self.apm
        return

    def subtract_background(self, data):
        """--------------------------------------------------------------------
        Procedure: Estimates and subtracts a uniform background

        !! The data is modified in place !!

        Parameters:
        ----------
        - data: 2D image or 3D data cube
        --------------------------------------------------------------------"""
        if self.bgm is None:
            self.make_apodizing_mask()  # default apodizing + background mask
        if data.ndim == 2:
            data -= np.median(data[self.bgm])
        elif data.ndim == 3:
            for ii in range(data.shape[0]):
                data[ii] -= np.median(data[ii][self.bgm])

    def sort_full_model(self, dense, sparse, frad0=0.5):
        """--------------------------------------------------------------------
        Sort the full (dense) model sample points coordinates into patches of
        sub-apertures.

        Apertures within a radius =rad0= of the sparse model are regrouped
        together. The function returns a 6-element list of indices within the
        dense model that are part of the same Fourier splodge (that should be
        centered on a point of the sparse model).
        --------------------------------------------------------------------"""
        uu0, vv0 = sparse.kpi.UVC.T
        uu1, vv1 = dense.kpi.UVC.T
        uu0, vv0 = np.append(uu0, -uu0), np.append(vv0, -vv0)
        uu1, vv1 = np.append(uu1, -uu1), np.append(vv1, -vv1)

        spl_ii = []  # splodge coordinate indices
        for ii in range(2 * sparse.kpi.nbuv):
            spl_ii.append(
                np.argwhere(
                    (np.abs(uu1-uu0[ii]) < frad0) * \
                    (np.abs(vv1-vv0[ii]) < frad0))[:,0])
        return spl_ii

    def compute_Fourier_slope_aux_data(self):
        """--------------------------------------------------------------------
        Compute reusable auxilliary data useful to estimate phase slope across
        a Fourier splodge.
        --------------------------------------------------------------------"""
        uu1, vv1 = self.K1DM.kpi.UVC.T
        uu1, vv1 = np.append(uu1, -uu1), np.append(vv1, -vv1)

        # use the first Fourier splodge as a template
        iis = self.spl_ii[0]
        self.uusl0, self.vvsl0 = uu1[iis], vv1[iis]
        self.uusl0 -= self.uusl0.mean()
        self.vvsl0 -= self.vvsl0.mean()
        self.uusl0 /= self.uusl0.dot(self.uusl0)
        self.vvsl0 /= self.vvsl0.dot(self.vvsl0)

    def get_splodge_slope(self, phase):
        """--------------------------------------------------------------------
        Returns the (u,v) slope of the Fourier phase values across a sploge
        --------------------------------------------------------------------"""
        du = phase.dot(self.uusl0)
        dv = phase.dot(self.vvsl0)
        return np.array([du, dv])

    def get_splodge_slopes(self, phase):
        """--------------------------------------------------------------------
        Returns the (u,v) slopes of all Fourier splodges
        --------------------------------------------------------------------"""
        nbuv = self.nbl
        duv = np.zeros((nbuv, 2))
        for ii in range(nbuv):
            duv[ii] = self.get_splodge_slope(phase[self.spl_ii[ii]])
        return duv

    def get_global_slope(self, phase):
        """--------------------------------------------------------------------
        Returns the (u,v) slopes of all Fourier splodges.

        !!! TBD !!!
        --------------------------------------------------------------------"""
        uu1, vv1 = self.K1DM.kpi.UVC.T
        uu1, vv1 = np.append(uu1, -uu1), np.append(vv1, -vv1)
        nbuv = self.nbl
        # for ii in range(nbuv):
        pass

    def get_splodge_pos(self, v2, coarse=False):
        """--------------------------------------------------------------------
        Returns the (u,v) v2 center of gravity of the splodges

        Parameters:
        ----------
        - v2: the (dense model) square visibility vector
        --------------------------------------------------------------------"""
        nbuv = self.nbl
        uv0 = np.zeros((nbuv, 2))

        uu, vv = self.dense.kpi.UVC.T
        uu, vv = np.append(uu, -uu), np.append(vv, -vv)

        if coarse:
            for ii in range(nbuv):
                imax = np.argmax(v2[self.spl_ii[ii]])  # index of max v2
                uv0[ii] = (uu[self.spl_ii[ii]][imax],
                           vv[self.spl_ii[ii]][imax])
        else:
            for ii in range(nbuv):
                uv0[ii] = center_of_mass_2D(
                    uu[self.spl_ii[ii]],
                    vv[self.spl_ii[ii]],
                    v2[self.spl_ii[ii]])
        return uv0

    def infer_pupil_model(self, v2, coarse=False):
        """--------------------------------------------------------------------
        Returns an updated set of pupil beam coordinates matching the v2

        Parameters:
        ----------
        - model: the KPI (sparse or dense) model to use for the extraction
        - img: the image to extract from
        - full: (bool) if True, appends the complex conjugated cvis

        Returns:
        -------
        - xy0: updated coordinates for the aperture model
        --------------------------------------------------------------------"""
        uv0 = self.get_splodge_pos(v2, coarse=coarse)
        xy0 = np.round(np.tensordot(self.iBLM, uv0, axes=(1, 0)), 3)
        xy0 -= xy0[2,:]  # beam #3 is our reference
        return xy0

    def get_pupil_wft(self, img, pfilter=True):
        """--------------------------------------------------------------------
        Analyzes an image and returns a wavefront, according to the dense model

        Parameters:
        ----------
        - img: a 32x32 interferogram image
        --------------------------------------------------------------------"""

        if pfilter:
            phi = np.angle(self.get_raw_cvis(self.dense, img, full=True))
            phi0 = np.angle(self.get_raw_cvis(self.sparse, img, full=True))

            for ii in range(2 * self.nbl):
                phi[self.spl_ii[ii]] -= phi0[ii]
                phi[self.spl_ii[ii]] = (phi[self.spl_ii[ii]] + 1.6) % (2*np.pi) - 1.6
            wft = np.append(0, self.dense.PINV.dot(phi[:396]))
        else:
            cvis = self.get_raw_cvis(self.dense, img, full=False)
            phi = np.angle(cvis)
            wft = np.append(0, self.dense.PINV.dot(phi))
        return wft

    def wft_to_ttxy(self, wft):
        """--------------------------------------------------------------------
        Converts a wavefront vector into tip-tilt components on the 4 beams

        Parameters:
        ----------
        - wft: a wavefront (result of self.get_pupil_wft(img))
        --------------------------------------------------------------------"""
        ttx, tty = np.zeros(self.nbm), np.zeros(self.nbm)
        for ii in range(self.nbm):
            swft = wft[ii*self.nbsap:(ii+1)*self.nbsap]
            ttx[ii], _ = np.polyfit(self.hexc[0,:], swft, 1) / np.pi**2
            tty[ii], _ = np.polyfit(self.hexc[1,:], swft, 1) / np.pi**2
        return ttx, tty

    def optimize_pupil_model(self, img):
        """--------------------------------------------------------------------
        Optimizes the pupil model on the basis of the analysis of the image
        --------------------------------------------------------------------"""
        # coarse iteration
        cvis = self.get_raw_cvis(self.dense, img)
        vis2 = np.abs(cvis)**2
        xyc = self.infer_pupil_model(vis2, coarse=True)
        self.update_pupil_model(hc_sparse=xyc)

        # fine iteration
        cvis = self.get_raw_cvis(self.dense, img)
        vis2 = np.abs(cvis)**2
        xyc = self.infer_pupil_model(vis2, coarse=False)
        self.update_pupil_model(hc_sparse=xyc)

    def get_raw_cvis(self, model, img, full=True):
        """--------------------------------------------------------------------
        Returns the raw complex visibility extracted by the model

        Parameters:
        ----------
        - model: the KPI (sparse or dense) model to use for the extraction
        - img: the image to extract from
        - full: (bool) if True, appends the complex conjugated cvis
        --------------------------------------------------------------------"""
        cvis = model.extract_cvis_from_img(img)
        if len(cvis) > 6:
            cvis /= self.nbsap  # dense model - vis / 19 to be within [0, 1]
        cvis = np.append(cvis, cvis.conj()) if full is True else cvis
        return cvis

    def plot_fourier_splodges(self, data, phase=False, cmap=cm.rainbow,
                              vmin=None, vmax=None):
        """--------------------------------------------------------------------
        Produces a 2D plot of the distribution of *data* according to the model.

        Parameters:
        ----------
        - data: the 1D vector of data to be plotted (|V|, |V|^2 or phase)
        - phase: flag to confirm whether the data is phase or not
        - cmap: a color map (default: rainbow)
        - vmin: (optional) minimum value clipping
        - vmax: (optional) maximum value clipping

        Returns: (fig, ax)
        --------------------------------------------------------------------"""
        uu1, vv1 = self.dense.kpi.UVC.T
        uu0, vv0 = self.sparse.kpi.UVC.T
        uu1, vv1 = np.append(uu1, -uu1), np.append(vv1, -vv1)

        vmin = data.min() if vmin is None else vmin
        vmax = data.max() if vmax is None else vmax

        fig, ax = plt.subplots()
        fig.set_size_inches(5, 5, forward=True)
        for ii in range(self.nbl * 2):
            iis = self.spl_ii[ii]
            ax.scatter(uu1[iis], vv1[iis], c=data[iis],
                       vmin=vmin, vmax=vmax, cmap=cmap)
        for ii in range(self.nbl):
            circ = plt.Circle((uu0[ii], vv0[ii]), radius=self.frad,
                              color=palette6[ii], fill=False, lw=4)
            ax.add_patch(circ)
            ax.text(uu0[ii], vv0[ii] + 1.1 * self.frad, self.bl_lbls[ii], ha="center")
        ax.axis('equal')
        ax.grid(True)
        return fig, ax

    def plot_histograms(self, vis2, bins=20, title=r"V$^2$ histograms",
                        alpha=0.7):
        """--------------------------------------------------------------------
        Produces a figure with histograms of the provided |V|^2 for
        all six baselines.

        Parameters:
        ----------
        - vis2: the (n x 6) array of visibilities to represent

        Returns: (fig, ax)
        --------------------------------------------------------------------"""
        fig, ax = plt.subplots(3, 2)
        print(title, "\n"+len(title)*"=")
        for ii in range(self.nbl):
            v2 = vis2[:,ii]
            lbl = rf"V$^2$ = {np.mean(v2):.2f} +/- {np.std(v2):.3f}"
            print(self.bl_lbls[ii], " - ", lbl)
            ax[ii % 3, ii // 3].hist(v2, color=palette6[ii], alpha=alpha,
                                       bins=bins, label=lbl)
            ax[ii % 3, ii // 3].set_title(self.bl_lbls[ii])
            ax[ii % 3, ii // 3].legend(loc=1)
        fig.set_tight_layout(True)
        fig.set_size_inches(8, 8, forward=True)
        fig.suptitle(title)
        return fig, ax

    def plot_cp_histograms(self, cp1, cp2, bins=20, cmax=0.5,
                           title="Closure-phase histograms", alpha=0.7):
        """--------------------------------------------------------------------
        Produces a figure with histograms of the provided closure-phases.

        Expects to get K1 and K2 closure-phases in sequence

        Parameters:
        ----------
        - K1: the 

        Returns: (fig, ax)
        --------------------------------------------------------------------"""
        fig, ax = plt.subplots(self.ncp, 2)
        for ii in range(self.ncp):
            lbl = rf"K$_1$ CP = {np.mean(cp1[ii]):.2f} +/- {np.std(cp1[ii]):.3f}"
            tax = ax[ii % self.ncp, 0]  # this ax
            tax.hist(
                cp1[ii], color=palette6[ii], range=[-cmax, cmax],
                alpha=0.5, bins=20, label=lbl)
            tax.set_title(self.cp_lbls[ii])
            tax.legend(loc=1)
            tax.grid(True, axis='x')

            lbl = rf"K$_2$ CP = {np.mean(cp2[ii]):.2f} +/- {np.std(cp2[ii]):.3f}"
            tax = ax[ii % self.ncp, 1]
            tax.hist(
                cp2[ii], color=palette6[ii], range=[-cmax, cmax],
                alpha=0.5, bins=20, label=lbl)
            tax.set_title(self.cp_lbls[ii])
            tax.legend(loc=1)
            tax.grid(True, axis='x')

        fig.set_tight_layout(True)
        fig.set_size_inches(8, 10, forward=True)
        fig.suptitle(title)
        return fig, ax

    def plot_line_plots(self, data, title="Fourier Phase", alpha=1.0):
        """--------------------------------------------------------------------
        Produces a figure with plots of the provided *data* for all baselines.

        Parameters:
        ----------
        - data: the (n x 6) array of visibilities to represent

        Returns: (fig, ax)
        --------------------------------------------------------------------"""
        fig, ax = plt.subplots(3, 2)
        print(title, "\n"+len(title)*"=")
        for ii in range(self.nbl):
            ax[ii % 3, ii // 3].plot(
                data[:, ii], color=palette6[ii], alpha=alpha)
            ax[ii % 3, ii // 3].set_title(self.bl_lbls[ii])
            print(self.bl_lbls[ii],
                  f"{np.mean(data[:, ii]):+5.2f} +/- {np.std(data[:,ii]):.3f}")
        fig.set_tight_layout(True)
        fig.set_size_inches(8, 8, forward=True)
        fig.suptitle(title)
        return fig, ax

    def extract_cube_cvis(self, dcube, model):
        """--------------------------------------------------------------------
        Extract the useful cvis from a data-cube in accordance with a model
        --------------------------------------------------------------------"""
        nim = dcube.shape[0]
        cvis = np.zeros((nim, 2 * model.kpi.nbuv), dtype=complex)
        for ii in range(nim):
            cvis[ii] = self.get_raw_cvis(model, dcube[ii])
        return cvis

    def fit_phase_slope(self, phi, xyc):
        """--------------------------------------------------------------------
        Fit a phase slope to the provided phase data, using the provided
        coordinates

        Parameters:
        ----------
        - phi: the Fourier phase data         - shape (nim * nbl)
        - xyc: the (x,y) or (u,v) coordinates - shape: (nbl * 2)

        Returns: a pure slope phase model of the original data
        --------------------------------------------------------------------"""
        xynorm = np.linalg.norm(xyc, axis=0)**2

        if phi.ndim == 2:
            slope = np.tensordot(phi, xyc, axes=(1, 0)) / xynorm
            phi_fit = np.tensordot(slope, xyc, axes=(1,1))
            return phi_fit
        else:
            # for RT single image processing
            print("later")
            pass
