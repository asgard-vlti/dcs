""" ====================================================================
The intention here is to have a live display of the temporal power
spectrum of the phase measured by Heimdallr.
==================================================================== """
import numpy as np
import pyqtgraph as pg

from xaosim.QtMain import QtMain
from xaosim.shmlib import shm

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRect

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

from wfs import Heimdallr

import threading

import time
import sys

# =====================================================================
#                   global variables and tools
# =====================================================================

myqt = 0  # myqt is a global variable
nbuv = 6  # the number of baselines to track for

# electric rainbow burst (ndg, blu, grn, ylw,rng, red)
palette6 = [(100, 52, 233), (44,124,229), (73,204,92),
            (248,196,33), (251,102,64), (248,37,83)]

# tropical summer vibes (dgrn, trk, grn, ylw, orng, dorng)
palette6 = [(38, 70, 83), (42,157,143), (138,177,125),
            (233,196,106), (244,162,97), (231,111,81)]

fps = 1000
ssize = 2000  # sample size

def power_spectrum(data):
    tmp = np.abs(np.fft.rfft(np.unwrap(data)))**2
    # tmp = np.abs(np.fft.rfft(data))**2
    # tmp = np.abs(np.fft.fft(data)[:1000])**2
    # return np.sqrt(tmp) / 2000 * np.sqrt(2)
    return tmp[1:] / ssize**2 * 2

freqs = np.fft.fftfreq(ssize, 1/fps)[:ssize//2]

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
        self.title = 'Asgard Lab Vibration Monitor'
        self.left, self.top = 0, 0
        self.width, self.height = 1200, 800
        self.band = "K1"  # default band for fringe search

        self.setWindowTitle(self.title) 
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setMinimumSize(QtCore.QSize(self.width, self.height))
        self.setMaximumSize(QtCore.QSize(self.width, self.height))
        self.main_widget = MyMainWidget(self) 
        self.setCentralWidget(self.main_widget) 

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(500)

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
        self.actionOpen = QtWidgets.QAction(
            QtGui.QIcon(":/images/open.png"), "&Open...", self)

        self.actionQuit = QtWidgets.QAction(
            QtGui.QIcon(":/images/open.png"), "&Quit", self)

        self.actionQuit.triggered.connect(self.close_program)
        self.actionQuit.setShortcut('Ctrl+Q')

        self.menu = parent.menuBar()
        file_menu = self.menu.addMenu("&File")
        file_menu.addAction(self.actionQuit)

        self.gView_plots = []
        for ii in range(nbuv):
            self.gView_plots.append(pg.PlotWidget(self))

        self.pB_start = QtWidgets.QPushButton(self)
        self.pB_start.setText("START")

        self.pB_stop = QtWidgets.QPushButton(self)
        self.pB_stop.setText("STOP")

        self.data_setup()
        self.apply_layout()

        self.tracking = False

    # =========================================================================
    def wfs_start(self):
        if self.tracking:
            print("Already tracking")
        else:
            self.tracking = True
            self.wfsThread = GenericThread(self.wfs.loop)
            self.wfsThread.start()

    # =========================================================================
    def wfs_stop(self):
        self.tracking = False
        self.wfs.stop()

    # =========================================================================
    def close_program(self):
        # called when using menu or ctrl-Q
        self.wfs_stop()
        try:
            self.wfs.close()
        except:
            pass
        sys.exit()

    # =========================================================================
    def data_setup(self):
        self.wfs = Heimdallr()

    # =========================================================================
    def apply_layout(self):
        clh = 28   # control line height
        plw = 500  # plot width
        plh = 250  # plot height
        btx = 2 * plw + 30  # buttons x position
        
        self.pB_start.setGeometry(QRect(btx, 30, 100, clh))
        self.pB_stop.setGeometry(QRect(btx, 60, 100, clh))

        # -------------------
        #  the live displays
        # -------------------
        px = 10 + np.array([0, 0, 0, plw, plw, plw])
        py = 10 + np.tile(np.arange(nbuv/2) * (10 + plh), 2).astype(int)

        for ii in range(nbuv):
            self.gView_plots[ii].setGeometry(QRect(px[ii], py[ii], plw, plh))
            self.gView_plots[ii].setYRange(-np.pi, np.pi)
            self.gView_plots[ii].setBackground('w')
            self.gView_plots[ii].showGrid(x=True, y=True, alpha=0.3)
            self.gView_plots[ii].setLogMode(False, True)
            self.gView_plots[ii].setYRange(-8, -1)
        self.logplot_phi = []
        self.logplot_cum = []
        self.lbls = []

        for ii in range(nbuv):
            self.logplot_phi.append(
                self.gView_plots[ii].plot([0, 250], [1, 1],
                pen=pg.mkPen(palette6[ii], width=2), name=f"phi #{ii+1}"))
            self.logplot_cum.append(
                self.gView_plots[ii].plot([0.5, 250], [1, 1],
                pen=pg.mkPen(palette6[ii], width=2), name=f"phi #{ii+1}"))
            self.lbls.append(
                pg.TextItem(f"20 Hz val = {ii}", color='k', fill='w',
                            anchor=(0.5, 0.5)))
            self.lbls[ii].setPos(200, -1)
            self.gView_plots[ii].addItem(self.lbls[ii])

        self.pB_start.clicked.connect(self.wfs_start)
        self.pB_stop.clicked.connect(self.wfs_stop)

    # =========================================================================
    def refresh_plot(self):
        if len(self.wfs.phi_k1[0]) == self.wfs.log_len:
            for ii in range(nbuv):
                pwr = power_spectrum(self.wfs.phi_k1[ii])
                # pwr = power_spectrum(self.wfs.vis_k1[ii])
                cum = np.cumsum(pwr[::-1])[::-1]
                self.logplot_phi[ii].setData(freqs, pwr)
                self.logplot_cum[ii].setData(
                    freqs, cum)
                ii0 = np.argwhere(freqs == 20)[0][0]
                val = np.round(cum[ii0], 3)
                self.lbls[ii].setText(f"20 Hz val = {val}")
                

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
