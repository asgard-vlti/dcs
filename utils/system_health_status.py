from xaosim.QtMain import QtMain

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRect

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

import threading

import time
import sys
import os

myqt = 0   # myqt is a global variable

services = ["CRED1 SERVER ",
            "DM SERVER    ",
            "SHARED MEMORY",
            "ASTRONOMER   "]

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
        self.title = 'System Health Status'
        self.left, self.top = 0, 0
        self.width, self.height = 400, 550

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
        pass

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

        self.lbl_IDs = []

        for service in services:
            self.lbl_IDs.append(service, self)

        # self.lbl_cred1.setText("CRED1 SERVER")
        self.apply_layout()

    # =========================================================================
    def apply_layout(self):
        clh = 28   # control line height
        plw = 600  # plot width
        plh = 250  # plot height
        btx = plw + 20  # buttons x position
        btx2 = plw + 140

        self.lbl_cred1.move(100, 150)
        #setGeometry(QRect(btx, 30, 100, clh))

    # =========================================================================
    def close_program(self):
        # called when using menu or ctrl-Q
        # self.dstream.close(erase_file=False)
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
