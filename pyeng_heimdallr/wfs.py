#!/usr/bin/env python3

import numpy as np
import sys
from xaosim.shmlib import shm
from xara.iwfs import IWFS
from xara.core import recenter
from xaosim.pupil import _dist as dist
import astropy.io.fits as pf
import time
import zmq
from datetime import datetime
import os

import asgard_alignment.controllino as co


# ----------------------------------------
# pupil geometry design
hcoords = np.loadtxt("N1_hole_coordinates.txt")

ddir = os.getenv('HOME') + '/fringe_monitor/'
if not os.path.exists(ddir):
    os.makedirs(ddir)

default_log = ddir + "log_fringe_monitor.log"
# ----------------------------------------
# piston mode design
dms = 12
dd = dist(dms, dms, between_pix=True)  # auxilliary array
tprad = 5.5  # the taper function radius
taper = np.exp(-(dd/tprad)**20)  # power to be adjusted ?
amask = taper > 0.4  # seems to work well

pst = np.zeros((dms, dms))
pst[amask] = 0.02  # 0.2 DM piston on GUI -> 10 ADU units

im_offset = 1000.0
ogain = 3.5  # DM optical gain (in microns / control unit)
oscale = 1e6 / (4*np.pi * ogain) # phase to microns conversion factor (* wl)

# ------------------------------------------------
# a simple unwrapping procedure for RT processing?
# ------------------------------------------------
def unwrap(val, prev):
    if np.abs(val - prev) < np.pi:
        return val
    elif val > prev:
        return val - 2*np.pi
    else:
        return val + 2*np.pi

def log(message="", logfile=default_log, echo=True):
    ''' -----------------------------------------------------------------------
    Simple logging utility to keep track of HFO and HPOL modulation sequences
    ----------------------------------------------------------------------- '''
    tstamp = datetime.utcnow().strftime('%D %H:%M:%S')
    myline = f"{tstamp}: {message}"
    with open(logfile, "a") as mylog:
        mylog.write(myline+'\n')
    if echo:
        print(myline)

class Heimdallr():
    # =========================================================================
    def __init__(self):
        self.ndm = 4  # number of DMs
        self.chn = 4  # channel number dedicated to Heimdallr
        self.xsz = 32
        self.dd = dist(self.xsz, self.xsz, between_pix=True)
        self.apod = np.exp(-(self.dd/10)**4)

        self.nbl = 6  # number of baselines
        self.ncp = 3  # number of closure-phases

        self.K1_norm = 100000.0  # used to fix the K1 photometry
        self.K2_norm = 100000.0  # used to fix the K2 photometry

        self.gd_offset = np.zeros(self.nbl)
        
        self.pscale = 35 * 1.85
        self.Ks_wl = 2.05e-6  # True Heimdallr Ks wavelength (in meters)
        self.Kl_wl = 2.25e-6  # True Heimdallr Kl wavelength (in meters)

        self.gd_factor = self.Kl_wl**2 / (self.Kl_wl - self.Ks_wl) * oscale
        self.pd1_factor = self.Ks_wl * oscale
        self.pd2_factor = self.Kl_wl * oscale

        self.hdlr1 = IWFS(array=hcoords)
        self.hdlr2 = IWFS(array=hcoords)

        # hard-coded closure-phase matrix
        self.CPM = np.array([[ 0,  0,  0,  1, -1, -1],
                             [ 1, -1,  0,  0,  0, -1],
                             [ 1,  0, -1, -1,  0,  0],
                             [-1,  1,  0,  0,  0,  1]])

        # phase and group delay to be measured relative to beam 3
        # so a custom PINV is requested here
        self.PINV = np.round(np.linalg.pinv(
            np.delete(self.hdlr1.kpi.BLM, 2, axis=1)), 2)

        # self.PINV = np.round(np.linalg.pinv(self.hdlr1.kpi.BLM), 2)

        self.hdlr1.update_img_properties(
            isz=self.xsz, wl=self.Ks_wl, pscale=self.pscale)
        self.hdlr2.update_img_properties(
            isz=self.xsz, wl=self.Kl_wl, pscale=self.pscale)

        self.Ks = shm("/dev/shm/hei_k1.im.shm", nosem=False)
        self.Kl = shm("/dev/shm/hei_k2.im.shm", nosem=False)
        self.semid = 7

        self.tracking_mode = "group"
        self.abort = False

        self.dms = []
        self.sems = []

        for ii in range(self.ndm):
            self.dms.append(shm(f"/dev/shm/dm{ii+1}disp{self.chn:02d}.im.shm"))
            self.sems.append(shm(f"/dev/shm/dm{ii+1}.im.shm", nosem=False))

        self.zmq_context = zmq.Context()
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 10000)
        self.socket.connect("tcp://192.168.100.2:5555")

        log("Starting fringe monitor!")
        # connecting to HFOs
        # ===================
        self.hfo_pos = np.zeros(self.ndm)
        print("---")
        for ii in range(self.ndm):
            self.hfo_pos[ii] = self.get_dl_pos(ii+1)
            log(f"HFO{ii+1} = {self.hfo_pos[ii]:8.2f} um")
        print("---")

        # connecting to HPOLs
        # ===================
        self.cc = co.Controllino("192.168.100.12", init_motors=False)
        self.hpol_IDs = [4,5,6,7]  # check first w/ Mike
        self.hpol_pos = np.zeros(self.ndm, dtype=int)
        for ii in range(self.ndm):
            self.hpol_pos[ii] = self.get_hpol_pos(ii+1)
            log(f"HPOL{ii+1} = {self.hpol_pos[ii]:+6d} (steps)")
        print("---")

        # ===================
        self.keepgoing = True
        self.log_len = 2000  # length of the sensor log
        self.init_logs()

        self.calibrated = False
        self.calibrating = False
        self.cloop_on = False
        self.disps = np.zeros(self.ndm)

        self.gain = 0.2

        self.bl_lbls = []
        for ii in range(self.nbl):
            bm1_id = np.argmax(self.hdlr1.kpi.BLM[ii]) + 1
            bm2_id = np.argmin(self.hdlr1.kpi.BLM[ii]) + 1
            self.bl_lbls.append(f"(B{bm1_id}-B{bm2_id})")
    # =========================================================================
    def init_logs(self):
        self.opds = [[], [], []]  # the log of the measured OPDs
        self.gdlays = [[], [], [], [], [], []]  # the log of group delays
        self.vis_k1 = [[], [], [], [], [], []]  # log of K1 visibility
        self.vis_k2 = [[], [], [], [], [], []]  # log of K2 visibility
        self.phi_k1 = [[], [], [], [], [], []]  # log of K1 phase
        self.phi_k2 = [[], [], [], [], [], []]  # log of K2 phase
        self.cp_k1 = [[], [], [], []]           # log of K1 closure-phase
        self.cp_k2 = [[], [], [], []]           # log of K2 closure-phase
        self.first_time = True

    # =========================================================================
    def calc_wfs_data(self):
        k1d = self.Ks.get_latest_data(self.semid).astype(float) - im_offset
        k2d = self.Kl.get_latest_data(self.semid).astype(float) - im_offset
        k1d *= self.apod
        k2d *= self.apod
        # img = recenter(data, verbose=False)
        self.norm1 = k1d.sum()
        self.norm2 = k2d.sum()

        if self.norm1 != 0:
            self.hdlr1.extract_data(k1d)
        if self.norm2 != 0:
            self.hdlr2.extract_data(k2d)

        # visibility scaling to photometry
        self.hdlr1.cvis[0] *= self.norm1 / self.K1_norm
        self.hdlr2.cvis[0] *= self.norm2 / self.K2_norm

        # memorizing previous state
        try:
            self._pd_prev_k1 = self._pd_k1.copy()
            self._pd_prev_k2 = self._pd_k2.copy()
            self._prev_gd_rad = self._gd_rad.copy()
            self.first_time = False
        except:
            self.first_time = True
            # print("First time")
            self.K1_norm = self.norm1
            self.K2_norm = self.norm2            
            # pass

        self._pd_k1 = np.angle(self.hdlr1.cvis[0])
        self._pd_k2 = np.angle(self.hdlr2.cvis[0])
        self._gd_rad = np.angle(self.hdlr1.cvis[0] * self.hdlr2.cvis[0].conj())

        if not self.first_time:  # real-time unwrapping of phase
            for ii in range(self.nbl):
                self._gd_rad[ii] = unwrap(self._gd_rad[ii],
                                          self._prev_gd_rad[ii])
                self._pd_k1[ii] = unwrap(self._pd_k1[ii],
                                          self._pd_prev_k1[ii])
                self._pd_k2[ii] = unwrap(self._pd_k2[ii],
                                          self._pd_prev_k2[ii])

        self._cp_now_k1 = self.CPM.dot(self._pd_k1)
        self._cp_now_k2 = self.CPM.dot(self._pd_k2)

        for ii in range(self.ncp):
            self._cp_now_k1[ii] = (self._cp_now_k1[ii] + 1) % (2*np.pi) - 1
            self._cp_now_k2[ii] = (self._cp_now_k2[ii] + 1) % (2*np.pi) - 1

        self.gdlay = self._gd_rad * self.gd_factor - self.gd_offset
        self.opd_now_k1 = self.PINV.dot(self._pd_k1 * self.pd1_factor)
        self.opd_now_k2 = self.PINV.dot(self._pd_k2 * self.pd2_factor)

        if self.tracking_mode == "phase1":
            self.dms_cmds = self.opd_now_k1
        elif self.tracking_mode == "phase2":
            self.dms_cmds = self.opd_now_k2
        else:  # self.tracking_mode == "group":
            self.dms_cmds = self.PINV.dot(self.gdlay)
            
        # self.dms_cmds = self.opd_now_k1  # (or k2) - as a test?
        # self.dms_cmds = np.insert(self.dms_cmds, 0, 0)
        # self.dms_cmds -= self.dms_cmds[2] # everything relative to Beam 3
        # print(f"\r{self.dms_cmds}", end="")

    # =========================================================================
    def log_opds(self):
        for ii in range(self.nbl):
            self.gdlays[ii].append(self.gdlay[ii])

        for ii in range(self.ndm-1):
            # self.opds[ii].append(self.opd_now_k1[ii])
            self.opds[ii].append(self.dms_cmds[ii])

        if len(self.opds[0]) > self.log_len:
            for ii in range(self.ndm-1):
                self.opds[ii].pop(0)

            for ii in range(self.nbl):
                self.gdlays[ii].pop(0)

    # =========================================================================
    def log_vis(self):
        for ii in range(self.nbl):
            self.vis_k1[ii].append(np.abs(self.hdlr1.cvis[0][ii]))
            self.vis_k2[ii].append(np.abs(self.hdlr2.cvis[0][ii]))
            self.phi_k1[ii].append(self._pd_k1[ii])
            self.phi_k2[ii].append(self._pd_k2[ii])

        for ii in range(self.ncp):
            self.cp_k1[ii].append(self._cp_now_k1[ii])
            self.cp_k2[ii].append(self._cp_now_k2[ii])

        if len(self.vis_k1[0]) > self.log_len:
            for ii in range(self.nbl):
                self.vis_k1[ii].pop(0)
                self.vis_k2[ii].pop(0)
                self.phi_k1[ii].pop(0)
                self.phi_k2[ii].pop(0)

            for ii in range(self.ncp):
                self.cp_k1[ii].pop(0)
                self.cp_k2[ii].pop(0)

    # =========================================================================
    def dispatch_opds(self):
        ref_beam = 0.25 * np.sum(self.dms_cmds)

        self.disps[0] = self.dms_cmds[0]
        self.disps[1] = self.dms_cmds[1]
        self.disps[2] = 0.0
        self.disps[3] = self.dms_cmds[2]

        if self.cloop_on:
            for ii in range(self.ndm):
                p0 = self.dms[ii].get_data()
                dm = 0.999 * (p0 + self.gain * self.disps[ii] * pst)
                self.dms[ii].set_data(dm)
                self.sems[ii].post_sems(1)

    # =========================================================================
    def get_hpol_pos(self, beamid=1):
        """ Get the HPOL stepper motor position for the requested beam ID # """
        return self.cc.where(self.hpol_IDs[beamid-1])

    # =========================================================================
    def move_hpol(self, pos, beamid=1):
        """ Move the HPOL stepper motor to *pos*  the requested beam ID # """
        self.cc.amove(self.hpol_IDs[beamid-1], pos)

    # =========================================================================
    def get_dl_pos(self, beamid=1):
        """ Get delay line position (HFO) for the requested beam ID # """
        self.socket.send_string(f"read HFO{beamid}")
        return float(self.socket.recv_string().strip()) * 1e3

    # =========================================================================
    def move_dl(self, pos, beamid=1):
        """ Move delay line (HFO) to *pos* for the requested beam ID # """
        self.socket.send_string(f"moveabs HFO{beamid} {1e-3 * pos:.5f}")
        self.socket.recv_string()  # acknowledgement

    # =========================================================================
    def fringe_search(self, beamid=1, srange=100.0, step=5.0, band="K1",
                      thorough=False):
        ''' -------------------------------------------------- 
        Fringe search!

        Parameters:
        ----------

        - beamid   : 1, 2, 3 or 4 (BEAM ID #) (int)
        - srange   : the +/- search range in microns (float)
        - step     : the scan step in microns (float)
        - band     : "K1" or "K2"
        - thorough : if False, stops faster after optimum found
        -------------------------------------------------- '''
        nav = 5  # number of measurements to average

        for ii in range(self.ndm):
            self.hfo_pos[ii] = self.get_dl_pos(ii+1)
            
        x0 = self.hfo_pos[beamid-1] # startup position
        # steps = np.arange(x0 - srange, x0 + srange, step)
        steps = np.linspace(x0 - srange, x0 + srange,
                            int(1+2*srange/step))
        sensor = self.hdlr1 if band == "K1" else self.hdlr2

        log(f"---- HFO{beamid} SCAN sequence starting ----")
        BLM = sensor.kpi.BLM.copy()
        bl_ii = np.argwhere(BLM[:, beamid-1] != 0)[:, 0]  # concerned BLines
        # the starting point
        best_pos = x0
        vis = []
        for jj in range(nav):
            vis.append(np.abs(sensor.cvis[0]))
        vis = np.mean(np.array(vis), axis = 0)
        uvis = np.round(np.abs(vis)[bl_ii], 2)  # "useful" visibilities
        best_vis = np.round(np.sqrt(np.mean(uvis**2)), 2)        
        print(f"HFO{beamid} x0  = {x0:8.2f}", end="")  # initial state
        print(uvis, best_vis)
        found_one = 0

        if self.cloop_on:
            self.cloop_on = False  # interrupting the loop
            print("Opening the loop")

        for ii, pos in enumerate(steps):
            self.move_dl(pos, beamid)
            logline = f"{band}-HFO{beamid} pos={pos:8.2f} "
            # print(f"HFO{beamid} pos = {pos:8.2f} ", end="")
            time.sleep(0.25)
            vis = []
            for jj in range(nav):
                vis.append(np.abs(sensor.cvis[0]))
            vis = np.mean(np.array(vis), axis = 0)

            uvis = np.round(vis[bl_ii], 2)  # the useful visibilities here
            global_vis = np.round(np.sqrt(np.mean(uvis**2)), 2)
            # print(uvis, global_vis, end="")

            # logging all 6 visibilities
            data2print = np.array2string(
                uvis, precision=2, floatmode='fixed', separator=', ')
            logline += f"vis={data2print}"
            log(logline)

            if (global_vis >= 1.01 * best_vis) and (global_vis > 0.15) :
                best_vis = global_vis
                best_pos = pos
                found_one += 1 # (found_one == 1) and
                print(f"    - Current best pos: {pos:.2f}")
            else:
                print()

            if self.abort is True:
                log(f"---- HFO{beamid} SCAN aborted ----")
                time.sleep(0.5)
                self.abort = False
                break

            if not thorough:
                if (global_vis < 0.8 * best_vis) and \
                   (pos > x0) and (best_vis > 0.15):
                    time.sleep(0.5)
                    break

        print(f"Done! Best pos is {best_pos:.2f} um for v = {best_vis:.2f}\n")
        print(f"The scan went from {steps[0]:.2f} um to {steps[-1]:.2f}\n")
        if thorough:
            print("Return to starting HFO position")
            self.move_dl(x0, beamid)
        else:
            self.move_dl(best_pos, beamid)
        time.sleep(2)
        return best_pos, best_vis

    # =========================================================================
    def dual_fringe_scan(self, beamid=1, srange=100.0, step=5.0,
                         logname=default_log):
        ''' -------------------------------------------------- 
        Fringe scan across provided range of HFO, recording 
        the response in both K1 and K2 bands. Called by the
        hpol_pos_scan()

        After the scan, returns the HFO back to its original
        position.

        Parameters:
        ----------

        - beamid   : 1, 2, 3 or 4 (BEAM ID #) (int)
        - srange   : the +/- search range in microns (float)
        - step     : the scan step in microns (float)
        -------------------------------------------------- '''
        nav = 5  # number of measurements to average

        for ii in range(self.ndm):
            self.hfo_pos[ii] = self.get_dl_pos(ii+1)
            
        x0 = self.hfo_pos[beamid-1] # startup position
        steps = np.linspace(x0 - srange, x0 + srange,
                            int(1+2*srange/step))

        log(f"---- HFO{beamid} SCAN sequence starting ----",
            logfile=logname)

        if self.cloop_on:
            self.cloop_on = False  # interrupting the loop
            print("Opening the loop")

        for ii, pos in enumerate(steps):
            msg1 = f"K1-HFO{beamid} pos={pos:8.2f} vis="
            msg2 = f"K2-HFO{beamid} pos={pos:8.2f} vis="
            self.move_dl(pos, beamid)            
            time.sleep(0.25)

            vis1, vis2 = [], []
            for jj in range(nav):
                vis1.append(np.abs(self.hdlr1.cvis[0]))
                vis2.append(np.abs(self.hdlr2.cvis[0]))

            vis1 = np.mean(np.array(vis1), axis = 0)
            vis2 = np.mean(np.array(vis2), axis = 0)

            # logging all 6 visibilities
            msg1 += np.array2string(
                vis1, precision=2, floatmode='fixed', separator=', ')
            msg2 += np.array2string(
                vis2, precision=2, floatmode='fixed', separator=', ')

            log(msg1, logfile=logname)
            log(msg2, logfile=logname)

            if self.abort is True:
                log(f"---- HFO{beamid} SCAN aborted ----",
                    logfile=logname)
                time.sleep(0.5)
                self.abort = False
                break

        log(f"---- End of HFO{beamid} SCAN ---", logfile=logname)
        print("Return to starting HFO position")
        self.move_dl(x0, beamid)
        time.sleep(2)

    # =========================================================================
    def hpol_pos_scan(self, beamid, pmin, pmax, srange=100.0, step=5.0):
        ''' -------------------------------------------------------------------
        HPOL optimal position procedure

        The idea is as follows: doing a ramp of HPOL commands.
        At each step, scan corresponding HFO and log output.

        At the end, bring back HFO and HPOL to their original spot.
        ------------------------------------------------------------------- '''
        ssize = 25  # step size for HPOL
        pos_all = np.linspace(pmin, pmax, int((pmax-pmin)/ssize+1))

        utcnow = datetime.utcnow()
        tstamp = utcnow.strftime("%Y%m%d_%H:%M:%S")
        logname = ddir + f"log_{tstamp}_HPOL{beamid}_scan.log"
        
        log(f"---- HPOL{beamid} SCAN sequence starting ----")
        x0_hpol = self.get_hpol_pos(beamid)  # initial position

        for pos in pos_all:
            self.move_hpol(pos, beamid)
            log(f"HPOL{beamid} pos = {pos}", logfile=logname)

            if self.abort is True:
                log(f"---- HPOL{beamid} SCAN aborted ----")
                time.sleep(0.5)
                self.abort = False
                break

            self.dual_fringe_scan(beamid=beamid, srange=srange,
                                  step=step, logname=logname)

        print(f"HPOL{beamid} back to initial position {x0_hpol}")
        self.move_hpol(x0_hpol, beamid)

    # =========================================================================
    def dm_modulation_response(self):
        ''' -------------------------------------------------------------------
        Sinusoidal DM modulation sequence to characterize PD/GD behavior
        ------------------------------------------------------------------- '''
        nmod = 100
        nc = [1, 2, 3, 4]
        a0 = 10.0 # 0.5
        nav = 10  # number of measures per modulation
        
        self.reset_dms()
        now = datetime.utcnow()

        # prepare sinusoidal modulation commands
        cmds = np.zeros((self.ndm, nmod))
        for jj in range(self.ndm):
            cmds[jj] = a0 * np.sin(np.linspace(0, nc[jj] * 2*np.pi, nmod))

        # cmds[0] *= 0.0
        # cmds[1] *= 0.0
        # cmds[3] *= 0.0
        
        # to store the data
        cvis1 = np.zeros((self.hdlr1.kpi.nbuv, nmod * nav), dtype=complex)
        cvis2 = np.zeros((self.hdlr2.kpi.nbuv, nmod * nav), dtype=complex)

        cube1 = np.zeros((nmod * nav, 32, 32))
        cube2 = np.zeros((nmod * nav, 32, 32))

        for ii in range(nmod):
            # modulate the DMs
            for jj in range(self.ndm):
                self.dms[jj].set_data(cmds[jj, ii] * pst)
                self.sems[jj].post_sems(1)
                time.sleep(0.05)

            # record the data
            for jj in range(nav):
                imK1 = self.Ks.get_latest_data() - im_offset
                imK2 = self.Kl.get_latest_data() - im_offset
                cube1[ii * nav + jj] = imK1
                cube2[ii * nav + jj] = imK2

                cvis1[:, ii * nav + jj] = self.hdlr1.extract_cvis_from_img(imK1)
                cvis2[:, ii * nav + jj] = self.hdlr2.extract_cvis_from_img(imK2)

        self.reset_dms()

        sdir = f"/home/asg/Data/{now.year}{now.month:02d}{now.day:02d}/custom/"
        if not os.path.exists(sdir):
            os.makedirs(sdir)
        
        fname_root = sdir+f"data_{now.hour:02d}:{now.minute:02d}:{now.second:02d}_"

        np.savetxt(fname_root + f"modulation_cvis1_a0={a0:.2f}.txt", cvis1)
        np.savetxt(fname_root + f"modulation_cvis2_a0={a0:.2f}.txt", cvis2)

        pf.writeto(fname_root + f"modulation_cube_k1_a0={a0:.2f}.fits", cube1,
                   overwrite=True)
        pf.writeto(fname_root + f"modulation_cube_k2_a0={a0:.2f}.fits", cube2,
                   overwrite=True)
        print("modulation test done")

    # =========================================================================
    def stop(self):
        self.keepgoing = False
        self.cloop_on = False
        time.sleep(0.1)
        self.init_logs()
        try:
            del self._pd_k1
            del self._pd_k2
            del self._gd_rad
        except:
            pass
        print("\n")

    # =========================================================================
    def loop(self):
        self.keepgoing = True
        # catch-up with the semaphore
        self.Ks.catch_up_with_sem(self.semid)

        # start the loop
        while self.keepgoing:
            self.calc_wfs_data()
            self.dispatch_opds()
            self.log_opds()
            self.log_vis()

    # =========================================================================
    def reset_dms(self):
        for ii in range(self.ndm):
            self.dms[ii].set_data(0.0 * pst)
            self.sems[ii].post_sems(1)

    # =========================================================================
    def close(self):
        self.Ks.close(erase_file=False)
        self.Kl.close(erase_file=False)
        self.socket.disconnect()
        # for ii in range(self.ndm):
        #     self.dms[ii].close(erase_file=False)
        #     self.sems[ii].close(erase_file=False)
