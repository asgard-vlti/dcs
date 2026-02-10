import numpy as np
import pyqtgraph as pg

from xaosim.QtMain import QtMain
from xaosim.shmlib import shm

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRect

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

from wfs import Heimdallr, log

import threading

import time
import sys
# import zmq

# =====================================================================
#                   global variables and tools
# =====================================================================

myqt = 0   # myqt is a global variable


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
        self.title = 'Asgard Lab Fringe Monitor'
        self.left, self.top = 0, 0
        self.width, self.height = 800, 810
        self.band = "K1"  # default band for fringe search
        self.delay2display = "Grp Delay"
        self.setWindowTitle(self.title) 
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setMinimumSize(QtCore.QSize(self.width, self.height))
        self.setMaximumSize(QtCore.QSize(self.width, self.height))
        self.main_widget = MyMainWidget(self) 
        self.setCentralWidget(self.main_widget) 

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

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

        # self.context = zmq.Context()
        # self.server = self.context.socket(zmq.REP)
        # self.server.bind("tcp://192.168.100.2:6661")
        # self.zmqThread = GenericThread(self.zmq_loop)
        # self.zmq_server_on = True
        # self.zmqThread.start()

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

        self.gView_plot_vis_k1 = pg.PlotWidget(self)
        self.gView_plot_vis_k2 = pg.PlotWidget(self)
        self.gView_plot_gdlay = pg.PlotWidget(self)

        self.pB_start = QtWidgets.QPushButton("START", self)
        self.pB_stop = QtWidgets.QPushButton("STOP", self)
        self.pB_cloop = QtWidgets.QPushButton("C-LOOP", self)
        self.pB_oloop = QtWidgets.QPushButton("O-LOOP", self)     
        self.pB_reset_dms = QtWidgets.QPushButton("RESET DMs", self)
        self.pB_set_phot = QtWidgets.QPushButton("SET PHOT", self)
        self.pB_gd_set_offset = QtWidgets.QPushButton("Offset GD", self)
        self.pB_gd_fgt_offset = QtWidgets.QPushButton("Reset GD", self)
        self.pB_dm_modul = QtWidgets.QPushButton("MOD-SEQ", self)

        # ----- scanning for fringes ----
        self.cmB_select_filter = QtWidgets.QComboBox(self)
        self.cmB_select_filter.addItem("K1")
        self.cmB_select_filter.addItem("K2")        
        
        self.dspB_scan_range = QtWidgets.QDoubleSpinBox(self)
        self.dspB_scan_range.setDecimals(0)
        self.srange_val = 100.0  # default scan range in microns
        self.scan_step = 5.0     # default scan step size in microns
        self.dspB_scan_range.setValue(self.srange_val)
        self.dspB_scan_range.setMinimum(5.0)
        self.dspB_scan_range.setMaximum(2000.0)
        
        self.chB_fine_scan = QtWidgets.QCheckBox(self)
        self.chB_fine_scan.setText("fine")
        
        self.pB_scan_beam1 = QtWidgets.QPushButton("SCAN HFO1", self)
        self.pB_scan_beam2 = QtWidgets.QPushButton("SCAN HFO2", self)
        self.pB_scan_beam3 = QtWidgets.QPushButton("SCAN HFO3", self)
        self.pB_scan_beam4 = QtWidgets.QPushButton("SCAN HFO4", self)

        self.pB_jump_pos = []
        self.pB_jump_neg = []
        for ii in range(4):
            self.pB_jump_pos.append(QtWidgets.QPushButton("JUMP+", self))
            self.pB_jump_neg.append(QtWidgets.QPushButton("JUMP-", self))

        # ----- HPOL control widgets -----
        self.pBs_scan_hpol = []
        self.hpol_val_lbl = []
        self.dspB_hpol_p0 = []  # scan start point
        self.dspB_hpol_p1 = []  # scan end point
        self.pB_hpol_jump_pos = []
        self.pB_hpol_jump_neg = []
        self.hpol_step = 25     # gives resolution ~ 0.5 degree

        for ii in range(4):
            self.pBs_scan_hpol.append(QtWidgets.QPushButton(self))
            self.pBs_scan_hpol[ii].setText(f"HPOL{ii+1}")
            self.dspB_hpol_p0.append(QtWidgets.QSpinBox(self))
            self.dspB_hpol_p1.append(QtWidgets.QSpinBox(self))
            # self.dspB_hpol_p0[ii].setDecimals(0)
            # self.dspB_hpol_p1[ii].setDecimals(0)
            self.dspB_hpol_p0[ii].setMinimum(-10000)
            self.dspB_hpol_p1[ii].setMinimum(-10000)
            self.dspB_hpol_p0[ii].setMaximum(10000)
            self.dspB_hpol_p1[ii].setMaximum(10000)
            # self.hpol_val_lbl.append(QLabel(self))

            self.pB_hpol_jump_pos.append(QtWidgets.QPushButton(self))
            self.pB_hpol_jump_neg.append(QtWidgets.QPushButton(self))
            self.pB_hpol_jump_pos[ii].setText("JUMP+")
            self.pB_hpol_jump_neg[ii].setText("JUMP-")

        # ----- phase/group delay display ----
        self.cmB_select_delay = QtWidgets.QComboBox(self)
        self.cmB_select_delay.addItem("Grp Delay")
        self.cmB_select_delay.addItem("Ph Delay1")
        self.cmB_select_delay.addItem("Ph Delay2")
        self.cmB_select_delay.addItem("Cl-phs K1")
        self.cmB_select_delay.addItem("Cl-phs K2")

        self.pB_abort = QtWidgets.QPushButton(self)
        self.pB_abort.setText("ABORT")

        self.data_setup()
        self.apply_layout()

        self.tracking = False

    # =========================================================================
    def zmq_loop(self):
        ''' Waiting for and processing (independant thread) ZMQ commands '''
        print("ZMQ business starting!")
        while self.zmq_server_on:
            pass
        print("ZMQ business is done")

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
    def wfc_start(self):
        if self.wfs.cloop_on is True:
            return
        self.wfs.cloop_on = True
        print("Closing the loop!")

    # =========================================================================
    def wfc_stop(self):
        self.wfs.cloop_on = False
        print("Breaking the loop!")
        
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
        self.wfs.tracking_mode = "group"

        # feeding the current hpol positions into the GUI
        for ii in range(4):
            self.dspB_hpol_p0[ii].setValue(self.wfs.hpol_pos[ii] - 100)
            self.dspB_hpol_p1[ii].setValue(self.wfs.hpol_pos[ii] + 100)

    # =========================================================================
    def apply_layout(self):
        clh = 28   # control line height
        plw = 500  # plot width
        plh = 250  # plot height
        btx = plw + 20  # buttons x position
        btx2 = plw + 140

        self.pB_start.setGeometry(QRect(btx, 30, 100, clh))
        self.pB_cloop.setGeometry(QRect(btx, 60, 100, clh))
        self.pB_oloop.setGeometry(QRect(btx, 90, 100, clh))
        self.pB_stop.setGeometry(QRect(btx, 120, 100, clh))

        self.pB_set_phot.setGeometry(QRect(btx + 140, 30, 100, clh))

        self.pB_reset_dms.setGeometry(QRect(btx + 140, 90, 100, clh))

        self.cmB_select_filter.setGeometry(QRect(btx, 230, 50, clh))
        self.dspB_scan_range.setGeometry(QRect(btx, 260, 60, clh))
        self.chB_fine_scan.setGeometry(QRect(btx + 70, 260, 50, clh))

        self.pB_scan_beam1.setGeometry(QRect(btx, 290, 100, clh))
        self.pB_scan_beam2.setGeometry(QRect(btx, 320, 100, clh))
        self.pB_scan_beam3.setGeometry(QRect(btx, 350, 100, clh))
        self.pB_scan_beam4.setGeometry(QRect(btx, 380, 100, clh))
        self.pB_abort.setGeometry(QRect(btx, 410, 100, clh))

        for ii in range(4):
            self.pB_jump_pos[ii].setGeometry(
                QRect(btx2, 290 + 30 * ii, 50, clh))
            self.pB_jump_neg[ii].setGeometry(
                QRect(btx2 + 50, 290 + 30 * ii, 50, clh))

        for ii in range(4):
            self.pBs_scan_hpol[ii].setGeometry(
                QRect(btx, 500+30*ii, 50, clh))
            self.dspB_hpol_p0[ii].setGeometry(
                QRect(btx + 50, 500+30*ii, 60, clh))
            self.dspB_hpol_p1[ii].setGeometry(
                QRect(btx + 110, 500+30*ii, 60, clh))
            self.pB_hpol_jump_pos[ii].setGeometry(
                QRect(btx + 170, 500+30*ii, 50, clh))
            self.pB_hpol_jump_neg[ii].setGeometry(
                QRect(btx + 220, 500+30*ii, 50, clh))

        # self.hpol_val_lbl = []


        self.pB_dm_modul.setGeometry(QRect(btx + 120, 750, 100, clh))

        self.cmB_select_delay.setGeometry(QRect(btx, 690, 100, clh))
        # apply/forget delay offsets
        self.pB_gd_set_offset.setGeometry(QRect(btx, 720, 100, clh))
        self.pB_gd_fgt_offset.setGeometry(QRect(btx, 750, 100, clh))
        
        # -------------------
        #  the live displays
        # -------------------
        py = 10 + np.arange(3) * (10 + plh)
        self.gView_plot_vis_k1.setGeometry(QRect(10, py[0], plw, plh))
        self.gView_plot_vis_k1.setYRange(0, 0.8)
        self.gView_plot_vis_k1.setBackground('w')
        self.gView_plot_vis_k1.showGrid(x=True, y=True, alpha=0.3)

        self.gView_plot_vis_k2.setGeometry(QRect(10, py[1], plw, plh))
        self.gView_plot_vis_k2.setYRange(0, 0.8)
        self.gView_plot_vis_k2.setBackground('w')
        self.gView_plot_vis_k2.showGrid(x=True, y=True, alpha=0.3)

        self.gView_plot_gdlay.setGeometry(QRect(10, py[2], plw, plh))
        self.gView_plot_gdlay.setBackground('w')
        self.gView_plot_gdlay.showGrid(x=True, y=True, alpha=0.3)

        self.logplot_vis_k1 = []
        self.logplot_vis_k2 = []
        self.logplot_gdlay = []

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        # electric rainbow burst (ndg, blu, grn, ylw,rng, red)
        palette6 = [(100, 52, 233), (44,124,229), (73,204,92),
                    (248,196,33), (251,102,64), (248,37,83)]

        # tropical summer vibes (dgrn, trk, grn, ylw, orng, dorng)
        palette6 = [(38, 70, 83), (42,157,143), (138,177,125),
                    (233,196,106), (244,162,97), (231,111,81)]

        for ii in range(self.wfs.nbl):
            self.logplot_vis_k1.append(self.gView_plot_vis_k1.plot(
                [0, self.wfs.log_len], [ii, ii],
                pen=pg.mkPen(palette6[ii], width=2), name=f"v2 #{ii+1}"))

        for ii in range(self.wfs.nbl):
            self.logplot_vis_k2.append(self.gView_plot_vis_k2.plot(
                [0, self.wfs.log_len], [ii, ii],
                pen=pg.mkPen(palette6[ii], width=2), name=f"v2 #{ii+1}"))

        # for ii in range(3):
        #     self.logplot_gdlay.append(self.gView_plot_gdlay.plot(
        #         [0, self.wfs.log_len], [ii, ii],
        #         pen=pg.mkPen(palette6[ii], width=2), name=f"OPD #{ii+1}"))

        for ii in range(self.wfs.nbl):
            self.logplot_gdlay.append(self.gView_plot_gdlay.plot(
                [0, self.wfs.log_len], [ii, ii],
                pen=pg.mkPen(palette6[ii], width=2), name=f"GD #{ii+1}"))

        self.pB_start.clicked.connect(self.wfs_start)
        self.pB_stop.clicked.connect(self.wfs_stop)
        self.pB_cloop.clicked.connect(self.wfc_start)
        self.pB_oloop.clicked.connect(self.wfc_stop)

        self.pB_set_phot.clicked.connect(self.set_photometry)
        self.pB_reset_dms.clicked.connect(self.reset_dms)

        self.cmB_select_filter.activated[str].connect(self.select_filter)
        self.select_filter()

        self.cmB_select_delay.activated[str].connect(self.select_delay)
        self.select_delay()

        self.dspB_scan_range.valueChanged.connect(self.update_scan_range)
        self.chB_fine_scan.stateChanged[int].connect(self.update_scan_step)
        self.pB_scan_beam1.clicked.connect(self.scan_beam1)
        self.pB_scan_beam2.clicked.connect(self.scan_beam2)
        self.pB_scan_beam3.clicked.connect(self.scan_beam3)
        self.pB_scan_beam4.clicked.connect(self.scan_beam4)
        self.pB_abort.clicked.connect(self.abort_action)

        for ii in range(4):
            self.pB_jump_pos[ii].clicked.connect(self.jump_HFO_pos(ii))
            self.pB_jump_neg[ii].clicked.connect(self.jump_HFO_neg(ii))

        self.pB_gd_set_offset.clicked.connect(self.set_gd_offset)
        self.pB_gd_fgt_offset.clicked.connect(self.fgt_gd_offset)

        for ii in range(4):
            self.pBs_scan_hpol[ii].clicked.connect(self.scan_hpol(ii))
            self.pB_hpol_jump_pos[ii].clicked.connect(self.jump_HPOL_pos(ii))
            self.pB_hpol_jump_neg[ii].clicked.connect(self.jump_HPOL_neg(ii))

        self.pB_dm_modul.clicked.connect(self.trigger_modulation)

    # =========================================================================
    def set_photometry(self):
        self.wfs.K1_norm = self.wfs.norm1
        self.wfs.K2_norm = self.wfs.norm2
        msg = f"Photometry: K1={self.wfs.K1_norm:.0f}, K2={self.wfs.K2_norm:.0f}"
        log(msg)

    # =========================================================================
    def scan_hpol(self, ii):
        def scan():
            self.hpol_scan_thread = GenericThread(
                self.wfs.hpol_pos_scan,
                beamid=ii+1, pmin=self.dspB_hpol_p0[ii].value(),
                pmax=self.dspB_hpol_p1[ii].value(),
                srange=self.srange_val, step=self.scan_step)
            self.hpol_scan_thread.start()
        return scan

    # =========================================================================
    def jump_HPOL_pos(self, ii):
        def jump():
            pos = self.wfs.get_hpol_pos(ii + 1)
            new_pos = pos + self.hpol_step
            log(f"HPOL{ii+1}: jump {pos} --> {new_pos}")
            self.wfs.move_hpol(new_pos, ii+1)
        return jump

    # =========================================================================
    def jump_HPOL_neg(self, ii):
        def jump():
            pos = self.wfs.get_hpol_pos(ii + 1)
            new_pos = pos - self.hpol_step
            log(f"HPOL{ii+1}: jump {pos} --> {new_pos}")
            self.wfs.move_hpol(new_pos, ii+1)
        return jump

    # =========================================================================
    def jump_HFO_pos(self, ii):
        def jump():
            pos = self.wfs.get_dl_pos(ii + 1)
            new_pos = pos + 2 * self.srange_val
            log(f"HFO{ii+1}: jump {pos:.2f} um --> {new_pos:.2f} um")
            self.wfs.move_dl(new_pos, ii+1)
        return jump

    # =========================================================================
    def jump_HFO_neg(self, ii):
        def jump():
            pos = self.wfs.get_dl_pos(ii + 1)
            new_pos = pos - 2 * self.srange_val
            log(f"HFO{ii+1}: jump {pos:.2f} um --> {new_pos:.2f} um")
            self.wfs.move_dl(new_pos, ii+1)
        return jump

    # =========================================================================
    def select_delay(self):
        self.delay2display = str(self.cmB_select_delay.currentText())
        if self.delay2display == "Ph Delay1":
            self.wfs.tracking_mode = "phase1"
        elif self.delay2display == "Ph Delay2":
            self.wfs.tracking_mode = "phase2"
        else:
            self.wfs.tracking_mode = "group"
        
        if "Cl-phs" in self.delay2display:
            for ii in range(self.wfs.ncp, self.wfs.nbl):
                self.logplot_gdlay[ii].setData(np.zeros(self.wfs.log_len))

    # =========================================================================
    def set_gd_offset(self):
        self.wfs.gd_offset = self.wfs.gdlay
        print("New set point!")

    # =========================================================================
    def fgt_gd_offset(self):
        self.wfs.gd_offset = np.zeros(self.wfs.nbl)
        print("Forgot set point!")

    # =========================================================================
    def select_filter(self):
        self.band = str(self.cmB_select_filter.currentText())
        # print(self.band)
        pass

    # =========================================================================
    def abort_action(self):
        self.wfs.abort = True
        pass

    # =========================================================================
    def update_scan_step(self):
        "Control of the scan step in microns"
        if self.chB_fine_scan.isChecked():
            self.scan_step = 0.5
        else:
            self.scan_step = 5.0

    # =========================================================================
    def trigger_modulation(self):
        print("Modulation experiment start")
        self.modulation_thread = GenericThread(
            self.wfs.dm_modulation_response)
        self.modulation_thread.start()

    # =========================================================================
    def update_scan_range(self):
        self.srange_val = self.dspB_scan_range.value()

    # =========================================================================
    def scan_beam(self, beamid=1):
        self.vscan_thread = GenericThread(
            self.wfs.fringe_search,
            beamid=beamid, srange=self.srange_val,
            step=self.scan_step, band=self.band)
        self.vscan_thread.start()

    # =========================================================================
    def scan_beam1(self):
        self.scan_beam(1)
        
    # =========================================================================
    def scan_beam2(self):
        self.scan_beam(2)

    # =========================================================================
    def scan_beam3(self):
        self.scan_beam(3)

    # =========================================================================
    def scan_beam4(self):
        self.scan_beam(4)
    
    # =========================================================================
    def reset_dms(self):
        self.wfs.reset_dms()
        log("Reset DM commands")

    # =========================================================================
    def refresh_plot(self):
        if self.delay2display == "Ph Delay1":
            for ii in range(self.wfs.nbl):
                self.logplot_gdlay[ii].setData(self.wfs.phi_k1[ii])

        elif self.delay2display == "Ph Delay2":
            for ii in range(self.wfs.nbl):
                self.logplot_gdlay[ii].setData(self.wfs.phi_k2[ii])

        elif self.delay2display == "Cl-phs K1":
            for ii in range(self.wfs.ncp):
                self.logplot_gdlay[ii].setData(self.wfs.cp_k1[ii])

        elif self.delay2display == "Cl-phs K2":
            for ii in range(self.wfs.ncp):
                self.logplot_gdlay[ii].setData(self.wfs.cp_k2[ii])

        else:
            for ii in range(self.wfs.nbl):
                self.logplot_gdlay[ii].setData(self.wfs.gdlays[ii])
            
        for ii in range(self.wfs.nbl):
            self.logplot_vis_k1[ii].setData(self.wfs.vis_k1[ii])
            self.logplot_vis_k2[ii].setData(self.wfs.vis_k2[ii])

# ==========================================================
# ==========================================================
def main():
    global myqt
    myqt = QtMain()

    gui = App()
    gui.show()
    myqt.mainloop()
    gui.main_widget.zmq_server_on = False
    myqt.gui_quit()


# ==========================================================
# ==========================================================
if __name__ == '__main__': 
    main()
