import numpy as np
import pyqtgraph as pg

from xaosim.QtMain import QtMain
from xaosim.shmlib import shm

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRect

import matplotlib.cm as cm

# from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

import threading

import time
import sys
import json

# =====================================================================
#                           global variables
# =====================================================================

myqt = 0  # myqt is a global variable


# =====================================================================
#                               Tools
# =====================================================================
def bound(val, low, high):
    return max(low, min(high, val))


def arr2im(arr, vmin=False, vmax=False, pwr=1.0, cmap=None, gamma=1.0):
    """--------------------------------------------------------
    convert 2D numpy array into image for display

    limits dynamic range, power coefficient and applies colormap
    --------------------------------------------------------"""
    arr2 = arr.astype("float")  # local array is modified

    mmin = np.percentile(arr2, 5) if vmin is False else vmin
    mmax = np.percentile(arr2, 99.9) if vmax is False else vmax
    mycmap = cm.magma if cmap is None else cmap

    arr2 -= mmin
    if mmax != mmin:
        arr2 /= mmax - mmin
    arr2 = arr2**pwr

    res = mycmap(arr2)
    res[:, :, 3] = gamma
    return res


# ============================================================
#                   Thread specifics
# ============================================================
class GenericThread(QtCore.QThread):
    """---------------------------------------------------
    generic thread class used to externalize the execution
    of a function (calibration, closed-loop) to a separate
    thread.
    ---------------------------------------------------"""

    def __init__(self, function, *args, **kwargs):
        QtCore.QThread.__init__(self)
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __del__(self):
        self.wait()

    def run(self):
        self.function(*self.args, **self.kwargs)
        return


# ==========================================================
#              Creating the Application
# ==========================================================
class App(QtWidgets.QMainWindow):
    # ------------------------------------------------------
    def __init__(self):
        super().__init__()
        self.title = "Asgard CRED1 viewer"
        self.left, self.top = 0, 0
        self.width, self.height = 850, 360

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
        self.main_widget.refresh()

    # ------------------------------------------------------
    def closeEvent(self, event):
        self.main_widget.close_program()

    # ------------------------------------------------------
    def mouseMoved(self, evt):
        pos = evt[0]
        if self.main_widget.gView_live.sceneBoundingRect().contains(pos):
            mousePoint = self.main_widget.gView_live.getPlotItem().vb.mapSceneToView(
                pos
            )
            self.main_widget.vline.setPos(mousePoint.x())
            self.main_widget.hline.setPos(mousePoint.y())
            yi, xi = int(mousePoint.y()), int(mousePoint.x())
            self.main_widget.pyi = bound(yi, 0, self.main_widget.imsize[0] - 1)
            self.main_widget.pxi = bound(xi, 0, self.main_widget.imsize[1] - 1)


# =============================================================================
# =============================================================================
class MyMainWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(QtWidgets.QWidget, self).__init__(parent)

        # ---------------------------------------------------------------------
        #                              top menu
        # ---------------------------------------------------------------------
        self.actionOpen = QtWidgets.QAction(
            QtGui.QIcon(":/images/open.png"), "&Open...", self
        )

        self.actionQuit = QtWidgets.QAction(
            QtGui.QIcon(":/images/open.png"), "&Quit", self
        )

        self.actionQuit.triggered.connect(self.close_program)
        self.actionQuit.setShortcut("Ctrl+Q")

        self.menu = parent.menuBar()
        file_menu = self.menu.addMenu("&File")
        file_menu.addAction(self.actionQuit)

        # ===============
        self.mySHM = shm("/dev/shm/cred1.im.shm")
        self.imsize = self.mySHM.get_data().shape[1:]  # data is a 3D cube
        with open("/home/asg/.config/cred1_split.json") as file:
            self.split_config = json.load(file)
        # ===============

        self.imw, self.imh = 2 * self.imsize[1], 2 * self.imsize[0]
        self.vmin = False
        self.vmax = False
        self.pwr = 1.0
        self.averaging = False
        self.NDMR = False
        self.mycmap = cm.jet
        self.pxi, self.pyi = 0, 0
        self.pxval = 0.0

        self.gView_live = pg.PlotWidget(self)

        self.lbl_stats = QtWidgets.QLabel(self)

        # ============= display controls =============
        self.chB_nonlinear = QtWidgets.QCheckBox("non-linear", self)
        self.chB_average = QtWidgets.QCheckBox("averaging", self)
        self.chB_NDMR = QtWidgets.QCheckBox("NDMR", self)

        self.dspB_disp_min = QtWidgets.QDoubleSpinBox(self)
        self.dspB_disp_max = QtWidgets.QDoubleSpinBox(self)
        self.dspB_disp_min.setDecimals(0)
        self.dspB_disp_max.setDecimals(0)
        self.dspB_disp_min.setMinimum(0)
        self.dspB_disp_min.setMaximum(65000)
        self.dspB_disp_max.setMinimum(0)
        self.dspB_disp_max.setMaximum(65000)
        self.dspB_disp_min.setValue(1000)
        self.dspB_disp_max.setValue(10000)

        self.chB_min = QtWidgets.QCheckBox("apply min", self)
        self.chB_max = QtWidgets.QCheckBox("apply max", self)

        self.cmB_cbar = QtWidgets.QComboBox(self)
        self.cmB_cbar.addItems(
            ["magma", "gray", "hot", "cool", "bone", "jet", "viridis"]
        )
        self.cmB_cbar.activated[str].connect(self.update_cbar)

        self.apply_layout()

    # =========================================================================
    def close_program(self):
        # called when using menu or ctrl-Q
        self.mySHM.close(erase_file=False)
        sys.exit()

    # =========================================================================
    def apply_layout(self):
        pad = 10  # border width
        clh = 28  # control line height
        btw = 100  # button width
        xcol1 = self.imw + 2 * pad
        xcol2 = xcol1 + btw + 5

        # -------------------
        #  the live displays
        # -------------------
        self.gView_live.setGeometry(QRect(pad, pad, self.imw, self.imh))
        self.gView_live.hideAxis("left")
        self.gView_live.hideAxis("bottom")
        self.imv_data = pg.ImageItem()
        self.overlay = pg.GraphItem()
        self.gView_live.addItem(self.imv_data)

        # stat display
        self.lbl_stats.setGeometry(QRect(self.imw + 2 * pad, 40, 180, 180))
        # cross-hair
        self.vline = pg.InfiniteLine(angle=90, movable=False)
        self.hline = pg.InfiniteLine(angle=0, movable=False)
        self.gView_live.addItem(self.vline, ignoreBounds=True)
        self.gView_live.addItem(self.hline, ignoreBounds=True)

        # boxes
        self.oboxes = []
        for roi in self.split_config:
            x0, y0 = self.split_config[roi]["x0"], self.split_config[roi]["y0"]
            if roi.startswith("baldr"):
                img_height = self.imsize[0]
                full_img_height = 256
                y0 -= (full_img_height-img_height)
            xsz, ysz = self.split_config[roi]["xsz"], self.split_config[roi]["ysz"]
            obox = pg.RectROI(
                [x0, y0],
                [xsz, ysz],
                pen="r",
                aspectLocked=True,
                movable=False,
                removable=False,
                resizable=False,
            )
            self.oboxes.append(obox)
            self.gView_live.addItem(obox)

        # ============= display controls =============
        y0 = 240
        self.chB_nonlinear.setGeometry(QRect(xcol1, y0, btw, clh))
        self.chB_nonlinear.stateChanged[int].connect(self.update_non_linear)

        self.chB_average.setGeometry(QRect(xcol2, y0, btw, clh))
        self.chB_average.stateChanged[int].connect(self.update_average_state)

        y0 = 255
        self.chB_NDMR.setGeometry(QRect(xcol2, y0, btw, clh))
        self.chB_NDMR.stateChanged[int].connect(self.update_NDMR_state)

        y0 = 270
        self.dspB_disp_min.setGeometry(QRect(xcol1, y0, btw, clh))
        self.dspB_disp_max.setGeometry(QRect(xcol1, y0 + 30, btw, clh))
        self.dspB_disp_min.valueChanged[float].connect(self.update_vmin)
        self.dspB_disp_max.valueChanged[float].connect(self.update_vmax)

        self.chB_min.setGeometry(QRect(xcol2, y0, btw, clh))
        self.chB_max.setGeometry(QRect(xcol2, y0 + 30, btw, clh))
        self.chB_min.stateChanged[int].connect(self.update_vmin_apply)
        self.chB_max.stateChanged[int].connect(self.update_vmax_apply)

        x0, y0 = pad, 300
        # color scale
        self.cmB_cbar.setGeometry(QRect(x0, y0, btw, clh))
        self.cmB_cbar.activated[str].connect(self.update_cbar)
        self.cmB_cbar.setCurrentIndex(0)
        self.update_cbar()

    # =========================================================
    def update_vmin_apply(self):
        if not self.chB_min.isChecked():
            self.vmin = False
        else:
            self.vmin = self.dspB_disp_min.value()

    # =========================================================
    def update_vmin(self):
        # self.vmin = False
        self.chB_min.setCheckState(True)
        # if self.chB_min.isChecked():
        self.vmin = self.dspB_disp_min.value()

    # =========================================================
    def update_vmax_apply(self):
        if not self.chB_max.isChecked():
            self.vmax = False
        else:
            self.vmax = self.dspB_disp_max.value()

    # =========================================================
    def update_vmax(self):
        # self.vmax = False
        self.chB_max.setCheckState(True)
        # if self.chB_max.isChecked():
        self.vmax = self.dspB_disp_max.value()

    # =========================================================
    def update_non_linear(self):
        self.pwr = 1.0
        if self.chB_nonlinear.isChecked():
            self.pwr = 0.25

    # =========================================================
    def update_average_state(self):
        if self.chB_average.isChecked():
            self.averaging = True
        else:
            self.averaging = False
        pass

    # =========================================================
    def update_NDMR_state(self):
        if self.chB_NDMR.isChecked():
            self.NDMR = True
        else:
            self.NDMR = False
        pass

    # =========================================================
    def test(self):
        pass
        print("Test success!")

    # =========================================================
    def update_cbar(self):
        cbar = str(self.cmB_cbar.currentText()).lower()
        try:
            exec("self.mycmap = cm.%s" % (cbar,))
        except AttributeError:
            self.mycmap = cm.jet

    # =========================================================
    def refresh(self):
        if self.averaging or self.NDMR:
            cube = self.mySHM.get_data(False, True)
            if self.averaging:
                self.data_img = cube.mean(0)
            elif self.NDMR:
                # Find the instances of 0 in the 1st pixels of each slice 
                last_frames = np.where(cube[:, 0, 2] == 0)[0]
                max_ix = np.max(cube[:, 0, 2])
                print(cube[:,0,2])
                first_frames = np.where(cube[:,0,2] == max_ix)[0]
                navg = np.min([len(last_frames), len(first_frames)])
                if navg > 0:
                    last_frames = last_frames[:navg]
                    first_frames = first_frames[:navg]
                    self.data_img = np.mean(cube[last_frames], axis=0) - np.mean(cube[first_frames], axis=0) + 1000
                else:
                    print("ERROR - setting NDMR mode average to first frame.")
                    self.data_img = cube[0]
        else:
            self.data_img = self.mySHM.get_latest_data_slice()
        self.refresh_img()
        self.refresh_stats()
        pass

    # =========================================================
    def refresh_img(self):
        self.imv_data.setImage(
            arr2im(
                self.data_img.T,
                vmin=self.vmin,
                vmax=self.vmax,
                pwr=self.pwr,
                cmap=self.mycmap,
            ),
            border=2,
        )
        if self.pxi < self.imsize[1] and self.pyi < self.imsize[0]:
            self.pxval = self.data_img[self.pyi, self.pxi]

    # =========================================================
    def refresh_stats(self, add_msg=None):

        pt_levels = [0, 5, 50, 90, 99, 100]
        pt_values = np.percentile(self.data_img, pt_levels)

        msg = "<pre>\n"

        self.mySHM.read_keywords()
        sz = self.mySHM.mtdata["size"]
        msg += f"imsize = {sz[0]:3d} x {sz[1]:3d} x {sz[2]:3d}"
        msg += f"\n\ncross-hair ({int(self.pxi):3d}, {int(self.pyi):3d})\n"
        msg += f"value = {self.pxval:8.1f}\n\n"

        # for ii, kwd in enumerate(self.mySHM.kwds):
        #    msg += f"{kwd['name']:10s} : {kwd['value']:10s} \n"

        for i, ptile in enumerate(pt_levels):
            msg += f"p-tile {ptile:3d} = {pt_values[i]:8.2f}\n"

        msg += f"img count  = {self.mySHM.get_counter():8d}\n"

        if add_msg is not None:
            msg += f"{add_msg}\n"
        msg += "</pre>"
        self.lbl_stats.setText(msg)


# ==========================================================
# ==========================================================
def main():
    global myqt
    myqt = QtMain()

    gui = App()
    gui.show()

    proxy = pg.SignalProxy(
        gui.main_widget.gView_live.scene().sigMouseMoved,
        rateLimit=10,
        slot=gui.mouseMoved,
    )

    myqt.mainloop()
    myqt.gui_quit()


# ==========================================================
# ==========================================================
if __name__ == "__main__":
    main()
