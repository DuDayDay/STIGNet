from PySide2 import QtWidgets, QtGui, QtCore
from pointcloud.pointshow import PointCloudWidget
from pointcloud.PointCloudPlayer import PointCloudPlayer_2
from OpenGL.GL import *
import cv2
from PySide2.QtUiTools import QUiLoader
class point_clip_play(QtWidgets.QMainWindow):
    def __init__(self):
        super(point_clip_play, self).__init__()
        loader = QUiLoader()
        self.ui = loader.load("ui/video_player_clip.ui")
        self.point_cloud_frame = []
        self.setCentralWidget(self.ui)
        self.setWindowTitle("点云片段播放")
        self.ui.pushButton.clicked.connect(self.clip_delete)
        self.points_cloud_widget = PointCloudWidget()  # 创建点云显示窗口
        self.points_cloud_widget.setParent(self.ui.openGLWidget)
        self.points_cloud_widget.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # 关闭时删除


    def points_frame(self, point_cloud, start, stop):
        self.point_cloud_frame = point_cloud
        self.start = start
        self.stop = stop
        self.play()
    def play(self):
        if self.point_cloud_frame:
            self.point_cloud_frame = self.point_cloud_frame[self.start:self.stop]
            print(self.point_cloud_frame)
            self.points_cloud_widget.show()
            self.points_cloud_widget.point_cloud_frame = self.point_cloud_frame
            self.player = PointCloudPlayer_2(self.points_cloud_widget, self)
            self.player.start(100)

    def clip_delete(self):
        self.close()