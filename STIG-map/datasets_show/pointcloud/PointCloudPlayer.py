from PySide2 import QtCore
import numpy as np

class PointCloudPlayer:
    def __init__(self, point_cloud_widget, main_window):
        self.point_cloud_widget = point_cloud_widget
        self.main_window = main_window
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_index = 0
        self.is_playing = False

    def start(self, interval=100):
        self.timer.start(interval)  # 设置更新间隔（毫秒）

    def stop(self):
        self.timer.stop()

    def update_frame(self):
        if self.point_cloud_widget.point_cloud_frame is not None and self.is_playing:
            # 更新当前帧索引
            # self.main_window.ui.horizontalSlider.setMaximum = (len(self.point_cloud_widget.point_cloud_frame) - 1)
            self.point_cloud_widget.frame_index = (self.point_cloud_widget.frame_index + 1) % len(self.point_cloud_widget.point_cloud_frame)
            self.point_cloud_widget.update()  # 请求重绘来实现
            self.main_window.ui.spinBox_4.setValue(self.point_cloud_widget.frame_index)
            self.main_window.ui.horizontalSlider.setValue(int(self.point_cloud_widget.frame_index))

    def update_frame_from_slider(self, value):
        self.frame_index = value
        self.point_cloud_widget.frame_index = value  # 更新当前帧
        print(value)
        self.main_window.ui.spinBox_4.setValue(value)
        self.main_window.ui.spinBox_3.setValue(value)
        # self.point_cloud_widget.update()  # 请求重绘

    def reset(self):
        self.main_window.ui.horizontalSlider.setValue(0)
        self.main_window.ui.spinBox_4.setValue(0)



class PointCloudPlayer_2:
    def __init__(self, point_cloud_widget, main_window):
        self.point_cloud_widget = point_cloud_widget
        self.main_window = main_window
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_index = 0
        self.centroid = None

    def start(self, interval=100):
        self.timer.start(interval)  # 设置更新间隔（毫秒）

    def stop(self):
        self.timer.stop()

    def update_frame(self):
        if self.point_cloud_widget.point_cloud_frame:
            # 如果这是第一帧，计算质心
            if self.centroid is None:
                first_frame = self.point_cloud_widget.point_cloud_frame[0]
                self.centroid = [sum(coord) / len(first_frame) for coord in zip(*first_frame)]

            # 更新当前帧索引
            self.point_cloud_widget.frame_index = (self.point_cloud_widget.frame_index + 1) % len(
                self.point_cloud_widget.point_cloud_frame)

            # 使用质心来平移当前帧
            current_frame = self.point_cloud_widget.point_cloud_frame[self.point_cloud_widget.frame_index]
            self.point_cloud_widget.center_point_cloud(current_frame, self.centroid)  # 将质心作为参数传入

            self.point_cloud_widget.update()

