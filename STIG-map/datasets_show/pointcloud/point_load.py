from PySide2 import QtWidgets
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QFileDialog, QMessageBox
from pointcloud.PCD import file_input, frame_divide, frame_save_2
from PySide2 import QtWidgets, QtCore
from pointcloud.PointCloudPlayer import PointCloudPlayer_2
from pointcloud.pointshow import PointCloudWidget
import sys
class PointLoad(QtWidgets.QMainWindow):
    point_cloud_loaded = QtCore.Signal(object, int, int, str )  # 定义信号

    def __init__(self,main_window):
        super(PointLoad, self).__init__()
        loader = QUiLoader()
        self.ui = loader.load('ui/point_load.ui', self)
        self.main_window = main_window
        self.setWindowTitle("点云视频设置")
        self.setCentralWidget(self.ui)

        self.ui.pointButton.clicked.connect(self.load_point)
        self.file_name = ""
        self.dict_name = ""
        self.resolution = 20000
        self.overlap = 0.5
        self.frame = 20

        self.ui.spinBox.valueChanged.connect(self.get_resolution)
        self.ui.spinBox_2.valueChanged.connect(self.get_overlap)
        self.ui.VideoButton_2.clicked.connect(self.get_point)
        self.ui.VideoButton_3.clicked.connect(self.get_point_show)
        self.ui.VideoButton_4.clicked.connect(self.save_pcd)
        self.point_cloud_frame = []

        self.player = None  # 初始化播放器
        self.point_cloud_widget = PointCloudWidget()  # 创建点云显示窗口
        self.point_cloud_widget.setParent(self.ui.openGLWidget)
        # self.point_cloud_widget.setGeometry(self.ui.openGLWidget.rect())  # 设置几何位置
        self.point_cloud_widget.setAttribute(QtCore.Qt.WA_DeleteOnClose)  # 关闭时删除
    def get_point(self):
        if self.file_name:
            self.point_cloud_frame = frame_divide(self.file_name, self.resolution, int(self.frame))
            self.point_cloud_loaded.emit(self.point_cloud_frame, int(self.resolution), int(len(self.point_cloud_frame)),self.dict_name)
            QMessageBox.information(self, "点云帧划分", f"设置成功:分辨率为{self.resolution}")
            self.close()
        else:
            QMessageBox.information(self,"错误","文件设置错误，请重新设置")

    def get_point_show(self):
        self.point_cloud_widget.show()
        if self.file_name:
            self.point_cloud_frame = frame_divide(self.file_name, self.resolution, int(self.frame))
            self.point_cloud_widget.point_cloud_frame = self.point_cloud_frame  # 传递点云数据
            self.point_cloud_widget.is_playing = True
            self.player = PointCloudPlayer_2(self.point_cloud_widget, self.main_window)
            self.player.start(100)  # 开始播放，每 100 毫秒更新一次

    def close_point_cloud_view(self):
        self.player.stop()  # 停止播放
        self.close_button.hide()  # 隐藏关闭按钮
        self.setCentralWidget(self.ui)  # 返回到原始 UI 界面
    def get_overlap(self):
        self.frame = self.ui.spinBox_2.value()

    def get_resolution(self):
        self.resolution = float(self.ui.spinBox.value())

    def load_point(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.Csv);;所有文件 (*)", options=options)
        if file_name:
            self.ui.label_4.setText(file_name)  # 显示文件名
            self.file_name = file_name

    def save_pcd(self):
        options = QtWidgets.QFileDialog.Options()
        directory_name = QtWidgets.QFileDialog.getExistingDirectory(self, "选择文件夹", "", options=options)
        self.dict_name = directory_name
        if directory_name:
            print(f"选择的文件夹: {directory_name}")
            frame_save_2(self.file_name, directory_name, self.resolution, self.frame)
            QMessageBox.information(self, "点云帧划分", f"保存成功，存放pcd文件的文件地址为{directory_name}")
        else:
            print("未选择任何文件夹")




