import numpy as np
import pyqtgraph as pg

from xaosim.QtMain import QtMain
from xaosim.shmlib import shm
from xaosim.pupil import _dist as dist
from xaosim.zernike import mkzer1

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRect

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

import threading

import time
import sys
import os

from hmd_tts import HMD_TTS

myqt = 0   # myqt is a global variable

os.environ["OPENBLAS_NUM_THREAD"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# ============================================================
#                   Thread specifics
# ============================================================
class GenericThread(QtCore.QThread):
    ''' ---------------------------------------------------
    generic thread class used to externalize the execution
    of a function (calibration, closed-loop) to a separate
    thread.
    --------------------------------------------------- '''
    def __init__(self, function, *args, **kwargs):
        QtCore.QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs
 
    def __del__(self):
        self.wait()
 
    def run(self):
        self.function(*self.args,**self.kwargs)
        return

# ==========================================================
#              Creating the Application
# ==========================================================
class App(QtWidgets.QMainWindow): 
    # ------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.title = 'Heimdallr TT tracker GUI'
        self.left, self.top = 0, 0
        self.width, self.height = 820, 550

        self.setWindowTitle(self.title) 
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setMinimumSize(QtCore.QSize(self.width, self.height))
        self.setMaximumSize(QtCore.QSize(self.width, self.height))
        self.main_widget = MyMainWidget(self) 
        self.setCentralWidget(self.main_widget) 

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)

    # ------------------------------------------------------
    def refresh(self):
        self.main_widget.refresh_plot()

    # ------------------------------------------------------
    def closeEvent(self, event):
        self.main_widget.close_program()

# =============================================================================
# =============================================================================
class MyMainWidget(QWidget):
    def __init__(self, parent): 
        super(QWidget, self).__init__(parent)

        # ---------------------------------------------------------------------
        #                              top menu
        # ---------------------------------------------------------------------
        self.log_len = 2000  # length of the sensor log
        self.actionOpen = QtWidgets.QAction(
            QtGui.QIcon(":/images/open.png"), "&Open...", self)

        self.actionQuit = QtWidgets.QAction(
            QtGui.QIcon(":/images/open.png"), "&Quit", self)

        self.actionQuit.triggered.connect(self.close_program)
        self.actionQuit.setShortcut('Ctrl+Q')

        self.gView_plot_ttx = pg.PlotWidget(self)
        self.gView_plot_tty = pg.PlotWidget(self)

        self.pB_start = QtWidgets.QPushButton("START", self)
        self.pB_stop = QtWidgets.QPushButton("STOP", self)
        self.pB_calibrate = QtWidgets.QPushButton("CALIB", self)
        self.pB_close_loop = QtWidgets.QPushButton("CLOSE-L", self)
        self.pB_open_loop = QtWidgets.QPushButton("OPEN-L", self)
        self.pB_reset_DMs = QtWidgets.QPushButton("RESET DMs", self)
        
        self.in_gain = QtWidgets.QLineEdit(self)
        self.in_wwf  = QtWidgets.QLineEdit(self)

        self.pB_setGain = QtWidgets.QPushButton("GAIN", self)
        self.pB_setWgt  = QtWidgets.QPushButton("WEIGHT", self)

        self.tracking = False
        self.close_loop_on = False

        self.data_setup()
        self.apply_layout()

    # =========================================================================
    def data_setup(self):

        self.ttx_log = [[], [], [], []]
        self.tty_log = [[], [], [], []]
        self.ttx_w = [0.1, 0.1, 0.1, 0.1]
        self.tty_w = [0.1, 0.1, 0.1, 0.1]

        self.first_time = True

        self.hmd = HMD_TTS()
        self.hmd.make_apodizing_mask(arad=10, pwr=4)
        self.semid = 7
        self.dstream = shm("/dev/shm/hei_k2.im.shm", nosem=False)

        img = self.dstream.get_data() * 1.0
        self.img = np.zeros_like(img)
        self.nav = 20  # number of averaged images
        self.hmd.subtract_background(img)
        self.hmd.optimize_pupil_model(img)

        self.gain = 0.01
        self.wwf = 0.1
        self.dms = []
        self.sems = []
        self.chn = 1  # DM test channel
        self.dmsz = 12  # DM linear size (in actuators)
        for ii in range(self.hmd.nbm):
            self.dms.append(shm(f"/dev/shm/dm{ii+1}disp{self.chn:02d}.im.shm"))
            self.sems.append(shm(f"/dev/shm/dm{ii+1}.im.shm", nosem=False))

        dd = dist(self.dmsz, self.dmsz, between_pix=True)  # auxilliary array
        taper = np.exp(-(dd/5.5)**20)  # power to be adjusted ?
        amask = taper > 0.4  # seems to work well

        self.tt_modes = np.zeros((2, self.dmsz, self.dmsz))
        self.tt_modes[1] = mkzer1(2, self.dmsz, 5)
        self.tt_modes[0] = mkzer1(3, self.dmsz, 5)

        for ii in range(2):
            self.tt_modes[ii] /= self.tt_modes[ii][amask].std()


        self.RESP = np.array([[10.71,  0.  ,  0.  ,  0.  , -4.51,  0.  ,  0.  ,  0.  ],
                              [ 0.  ,  9.36,  0.  ,  0.  ,  0.  , -1.35,  0.  ,  0.  ],
                              [ 0.  ,  0.  ,  9.79,  0.  ,  0.  ,  0.  , -2.5 ,  0.  ],
                              [ 0.  ,  0.  ,  0.  ,  3.34,  0.  ,  0.  ,  0.  ,  8.05],
                              [-2.68,  0.  ,  0.  ,  0.  , -8.7 ,  0.  ,  0.  ,  0.  ],
                              [ 0.  , -0.97,  0.  ,  0.  ,  0.  , -9.25,  0.  ,  0.  ],
                              [ 0.  ,  0.  , -0.23,  0.  ,  0.  ,  0.  , -2.03,  0.  ],
                              [ 0.  ,  0.  ,  0.  ,  7.07,  0.  ,  0.  ,  0.  , -7.2 ]])

        print("-----------------------")
        print(np.round(self.RESP, 2))
        print("Checksum   :", np.round(np.sqrt(np.sum(self.RESP**2, axis=1)),2))

        for ii in range(self.hmd.nbm):
            az = np.arctan(self.RESP[ii+self.hmd.nbm,ii] / self.RESP[ii,ii]) * 180/np.pi
            print(f"DM#{ii+1} - Azim = {az:.1f} deg")
        print("-----------------------")
        self.PINV = np.linalg.pinv(self.RESP)

    # =========================================================================
    def apply_layout(self):
        clh = 28   # control line height
        plw = 600  # plot width
        plh = 250  # plot height
        btx = plw + 20  # buttons x position
        btx2 = plw + 140

        self.pB_start.setGeometry(QRect(btx, 30, 100, clh))
        self.pB_stop.setGeometry(QRect(btx, 60, 100, clh))
        self.pB_calibrate.setGeometry(QRect(btx, 120, 100, clh))
        self.pB_close_loop.setGeometry(QRect(btx, 150, 100, clh))
        self.pB_open_loop.setGeometry(QRect(btx, 180, 100, clh))
        self.pB_reset_DMs.setGeometry(QRect(btx, 210, 100, clh))

        self.pB_setGain.setGeometry(QRect(btx+70, 270, 70, clh))
        self.pB_setWgt.setGeometry(QRect(btx+70, 300, 70, clh))

        self.in_gain.setGeometry(QRect(btx, 270, 60, clh))
        self.in_wwf.setGeometry(QRect(btx, 300, 60, clh))

        self.in_gain.setText(f"{self.gain}")
        self.in_wwf.setText(f"{self.wwf}")

        self.gView_plot_ttx.setGeometry(QRect(10, 10, plw, plh))
        self.gView_plot_tty.setGeometry(QRect(10, 30 + plh, plw, plh))
        self.gView_plot_ttx.setYRange(-0.5, 0.5)
        self.gView_plot_tty.setYRange(-0.5, 0.5)
        self.gView_plot_ttx.setBackground('w')
        self.gView_plot_tty.setBackground('w')
        self.gView_plot_ttx.showGrid(x=True, y=True, alpha=0.3)
        self.gView_plot_tty.showGrid(x=True, y=True, alpha=0.3)

        # tropical summer vibes (dgrn, trk, grn, ylw, orng, dorng)
        palette6 = [(38, 70, 83), (42,157,143), (138,177,125),
                    (233,196,106), (244,162,97), (231,111,81)]

        self.ttx_logplots = [] # handles on the individual line plots
        self.tty_logplots = [] # handles on the individual line plots

        for ii in range(self.hmd.nbm):
            self.ttx_logplots.append(self.gView_plot_ttx.plot(
                [0, self.log_len], [0.1 * ii, 0.1 * ii],
                pen=pg.mkPen(palette6[ii], width=2), name=f"ttx{ii+1}"))
            self.tty_logplots.append(self.gView_plot_tty.plot(
                [0, self.log_len], [0.1 * ii, 0.1 * ii],
                pen=pg.mkPen(palette6[ii], width=2), name=f"tty{ii+1}"))

        self.pB_start.clicked.connect(self.tracker_start)
        self.pB_stop.clicked.connect(self.tracker_stop)
        self.pB_calibrate.clicked.connect(self.trigger_calibrate)
        self.pB_close_loop.clicked.connect(self.trigger_close)
        self.pB_open_loop.clicked.connect(self.trigger_open)
        self.pB_reset_DMs.clicked.connect(self.trigger_reset)

        self.pB_setGain.clicked.connect(self.set_gain)
        self.pB_setWgt.clicked.connect(self.set_wwf)

    # =========================================================================
    def tracker_start(self):
        if self.tracking:
            print("Already tracking")
        else:
            self.tracking = True
            print("Start")
            self.tt_thread = GenericThread(self.loop)
            self.tt_thread.start()

    # =========================================================================
    def set_gain(self):
        try:
            gain = float(self.in_gain.text())
        except:
            print("numerical value required")
            self.in_gain.setText(f"{self.gain}")
            return
        if 0.1 > gain > 0:
            self.gain = gain
            print(f"set gain = {self.gain}")

        else:
            self.in_gain.setText(f"{self.gain}")

    # =========================================================================
    def set_wwf(self):
        try:
            wwf = float(self.in_wwf.text())
        except:
            print("numerical value required")
            self.in_wwf.setText(f"{self.wwf}")
            return
        if 0.2 > wwf > 0:
            self.wwf = wwf
            print(f"set wwf = {self.wwf}")
        else:
            self.in_wwf.setText(f"{self.wwf}")

    # =========================================================================
    def trigger_calibrate(self):
        print("Calibrating")
        self.cal_thread = GenericThread(self.calibrate_dms)
        self.cal_thread.start()

    # =========================================================================
    def trigger_close(self):
        if self.close_loop_on:
            print("Loop already closed")
        else:
            self.close_loop_on = True
            print("Loop closed")
    # =========================================================================
    def trigger_open(self):
        if not self.close_loop_on:
            print("Loop was already opened")
        else:
            self.close_loop_on = False
            print("Loop opened")

    # =========================================================================
    def trigger_reset(self):
        for ii in range(self.hmd.nbm):
            self.dms[ii].set_data(0 * self.tt_modes[0])  # reset
            self.sems[ii].post_sems(1)

    # =========================================================================
    def tracker_stop(self):
        self.tracking = False
        print("Done")

    # =========================================================================
    def refresh_plot(self):
        for ii in range(self.hmd.nbm):
            self.ttx_logplots[ii].setData(self.ttx_log[ii])
            self.tty_logplots[ii].setData(self.tty_log[ii])

    # =========================================================================
    def loop(self):
        # self.tracking = True
        self.dstream.catch_up_with_sem(self.semid)

        while self.tracking:
            self.ttx, self.tty = self.get_signal()
            self.log_data()
            if self.close_loop_on:
                cmd = self.PINV.dot(np.append(self.ttx, self.tty))
                self.dispatch(cmd)

    # =========================================================================
    def get_signal(self):
        img = np.zeros_like(self.img)
        for ii in range(self.nav):
            img += (self.dstream.get_data() * 1.0 - 1000.0)
        self.hmd.apodize_data(img/self.nav)
        wft = self.hmd.get_pupil_wft(img, pfilter=True)
        return self.hmd.wft_to_ttxy(wft)
            
    # =========================================================================
    def log_data(self):
        for ii in range(self.hmd.nbm):
            self.ttx_log[ii].append(self.ttx[ii])
            self.tty_log[ii].append(self.tty[ii])
            
        if len(self.ttx_log[0]) > self.log_len:
            for ii in range(self.hmd.nbm):
                self.ttx_log[ii].pop(0)
                self.tty_log[ii].pop(0)

                # weighting factors based on SNR of last 50 points
                self.ttx_w[ii] = self.wwf / np.var(self.ttx_log[ii][-50:])
                self.tty_w[ii] = self.wwf / np.var(self.ttx_log[ii][-50:])

        print(f"\r{np.round(self.ttx_w, 3)}", end='', flush=True)

    # =========================================================================
    def dispatch(self, cmd):
        for ii in range(self.hmd.nbm):
            corrx, corry = cmd[ii], cmd[ii+self.hmd.nbm]
            correc = self.ttx_w[ii] * corrx * self.tt_modes[0] + \
                self.tty_w[ii] * corry * self.tt_modes[1]
            # correc = corrx * self.tt_modes[0] + corry * self.tt_modes[1]
            dm0 = self.dms[ii].get_data()
            self.dms[ii].set_data(0.999 * (dm0 - self.gain * correc))
            self.sems[ii].post_sems(1)

    # =========================================================================
    def calibrate_dms(self, a0=0.01):
        self.trigger_reset()
        ref_ttx, ref_tty = self.get_signal()

        nbm = self.hmd.nbm
        ndof = 2 * nbm  # 8 degrees of freedom (4 beams, 2 axes)
        resp = np.zeros((ndof, ndof))

        for ii in range(nbm):
            dm0 = self.dms[ii].get_data()  # original state of channel

            # ----- ttx -----
            self.dms[ii].set_data(dm0 + a0 * self.tt_modes[0])
            self.sems[ii].post_sems(1)
            time.sleep(1.0)
            sig_ttx, sig_tty = self.get_signal()
            resp[ii, ii] = (sig_ttx[ii] - ref_ttx[ii])
            resp[ii+nbm, ii] = (sig_tty[ii] - ref_tty[ii])

            # ----- tty -----
            self.dms[ii].set_data(dm0 + a0 * self.tt_modes[1])
            self.sems[ii].post_sems(1)
            time.sleep(1.0)
            sig_ttx, sig_tty = self.get_signal()
            resp[ii, ii+nbm] = (sig_ttx[ii] - ref_ttx[ii])
            resp[ii+nbm, ii+nbm] = (sig_tty[ii] - ref_tty[ii])
            
            # ------ reset -----
            time.sleep(1.0)
            self.dms[ii].set_data(dm0)
            self.sems[ii].post_sems(1)

        self.RESP = np.round(resp / a0, 2)
        self.PINV = np.linalg.pinv(self.RESP)
        print(np.round(self.RESP, 2))

        print("Checksum:", np.sqrt(np.sum(self.RESP**2, axis=1)))
        for ii in range(nbm):
            az = np.arctan(self.RESP[ii+nbm,ii] / self.RESP[ii,ii]) *180/np.pi
            print(f"DM#{ii+1} - Azim = {az:.1f} deg")

    # =========================================================================
    def close_program(self):
        # called when using menu or ctrl-Q
        self.dstream.close(erase_file=False)
        # self.wfs_stop()
        # try:
        #     self.wfs.close()
        # except:
        #     pass
        sys.exit()


# ==========================================================
# ==========================================================
def main():
    global myqt
    myqt = QtMain()

    gui = App()
    gui.show()
    myqt.mainloop()
    myqt.gui_quit()


# ==========================================================
# ==========================================================
if __name__ == '__main__': 
    main()

