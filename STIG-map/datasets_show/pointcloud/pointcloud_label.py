import sys
from PyQt5 import QtWidgets
from labelCloud.control.controller import Controller
from labelCloud.view.gui import GUI


def  PointCloud_label():
    app = QtWidgets.QApplication(sys.argv)
    # Setup Model-View-Control structure
    control = Controller()
    view = GUI(control)
    # Install event filter to catch user interventions
    app.installEventFilter(view)
    # Start GUI
    view.show()
    return app, view


if __name__ == '__main__':
    app, _ =  PointCloud_label()
    sys.exit(app.exec_())
